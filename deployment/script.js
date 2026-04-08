let model;
const IMG_SIZE = 128;

// GraphModel input tensor names (from model signature)
const QUERY_INPUT = 'inputs:0';   // shape [?, 128, 128, 1]
const PROMPT_INPUT = 'inputs_1:0'; // shape [?, 128, 128, 2]

async function runDummyTest() {
    console.log("Running Dummy Test...");
    tf.tidy(() => {
        const dummyQuery = tf.zeros([1, 128, 128, 1]);
        const dummyPrompt = tf.zeros([1, 128, 128, 2]);

        const inputs = {};
        inputs[QUERY_INPUT] = dummyQuery;
        inputs[PROMPT_INPUT] = dummyPrompt;

        const result = model.execute(inputs);
        console.log("Dummy test success! Output shape:", result.shape);
    });
}

function updateStatus(isReady) {
    const status = document.getElementById('statusStat');
    if (isReady) {
        status.innerText = "Ready";
        status.className = "stat-value status-ready";
    } else {
        status.innerText = "Loading";
        status.className = "stat-value";
    }
}

async function init() {
    try {
        // Load model as GraphModel (exported via SavedModel bridge)
        model = await tf.loadGraphModel('./p_unet_282_tfjs/model.json');
        console.log("Model loaded successfully.");

        await runDummyTest();

        const btn = document.getElementById('predictBtn');
        btn.querySelector('.btn-text').innerText = 'Run Segmentation';
        btn.disabled = false;
        btn.querySelector('.loader').style.display = 'none';
        updateStatus(true);
    } catch (e) {
        console.error("Failed to load model:", e);
        document.getElementById('statusStat').innerText = "Load Failed";
        document.getElementById('statusStat').style.color = "#fc8181";
    }
}

function getGrayscaleTensor(imgElement) {
    return tf.tidy(() => {
        const rgb = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([IMG_SIZE, IMG_SIZE])
            .toFloat();
        const gray = rgb.mean(2).expandDims(2);
        return gray.div(255.0).expandDims(0); // [1, 128, 128, 1]
    });
}

document.getElementById('predictBtn').onclick = async () => {
    const imgRef = document.getElementById('imgRef');
    const imgMask = document.getElementById('imgMask');
    const imgTarget = document.getElementById('imgTarget');

    if (!imgRef.src || !imgMask.src || !imgTarget.src) {
        alert("Please upload all three input images.");
        return;
    }

    const btnBtn = document.getElementById('predictBtn');
    btnBtn.disabled = true;
    btnBtn.querySelector('.loader').style.display = 'block';

    // Allow UI to refresh before freezing thread
    await new Promise(resolve => setTimeout(resolve, 50));

    // Garbage collect explicitly if needed
    // tf.engine().endScope(); tf.engine().startScope();

    try {
        const { promptTensor, queryTensor } = tf.tidy(() => {
            const refMRI = getGrayscaleTensor(imgRef).squeeze(0);
            const userMask = getGrayscaleTensor(imgMask).squeeze(0);

            const pTensor = tf.concat([refMRI, userMask], 2).expandDims(0);
            const qTensor = getGrayscaleTensor(imgTarget);

            return { promptTensor: pTensor, queryTensor: qTensor };
        });

        const inputDict = {};
        inputDict[QUERY_INPUT] = queryTensor;
        inputDict[PROMPT_INPUT] = promptTensor;

        // --- INFERENCE AND BENCHMARKING ---
        const t0 = performance.now();
        // GraphModel uses execute(), and output may be float16 so cast to float32
        let prediction = model.execute(inputDict);
        prediction = prediction.cast('float32');

        // Wait for WebGL logic to sync if necessary via dataSync
        prediction.dataSync();

        const t1 = performance.now();
        const intTimeMs = (t1 - t0).toFixed(1);
        document.getElementById('inferenceStat').innerText = `${intTimeMs} ms`;

        // Update memory usage stats
        const memObj = tf.memory();
        document.getElementById('memoryStat').innerText = `${(memObj.numBytes / (1024 * 1024)).toFixed(2)} MB`;

        // --- OVERLAY DRAWING LOGIC ---
        const canvas = document.getElementById('outputCanvas');
        document.getElementById('canvasPlaceholder').style.display = 'none';

        const blendedResult = tf.tidy(() => {
            // 1. Expand query MRI to 3 channels [128, 128, 3]
            const basemap = queryTensor.squeeze(0).tile([1, 1, 3]);

            // 2. Fetch mask [128, 128, 1] over threshold 0.45
            const mask = prediction.squeeze().greater(0.45).toFloat().expandDims(2);

            // 3. Define Overlay Color: Vibrant Cyan [0.0, 1.0, 0.9]
            const overlayColor = tf.tensor1d([0.0, 1.0, 0.9]);

            // 4. Alpha Blending Configuration
            const overlayAlpha = 0.5;
            const maskAlphaT = mask.mul(overlayAlpha);

            // Equation: Image * (1 - maskAlpha) + ColorMask * maskAlpha
            // We only apply color where mask is 1
            const colorMask = mask.mul(overlayColor);

            const blended = basemap.mul(tf.scalar(1).sub(maskAlphaT)).add(colorMask.mul(maskAlphaT));
            return blended;
        });

        await tf.browser.toPixels(blendedResult, canvas);

        // --- DRAW PROMPT OVERLAY FOR COMPARISON ---
        const promptCanvas = document.getElementById('promptCanvas');
        document.getElementById('promptPlaceholder').style.display = 'none';

        const blendedPrompt = tf.tidy(() => {
            const refImg = getGrayscaleTensor(imgRef).squeeze(0).tile([1, 1, 3]);
            const pMask = getGrayscaleTensor(imgMask).squeeze().greater(0.5).toFloat().expandDims(2);


            // Re-using the same cyan for consistency and 1:1 visual ratio
            const overlayColor = tf.tensor1d([0.0, 1.0, 0.9]);
            const maskAlphaT = pMask.mul(0.5);

            const colorMask = pMask.mul(overlayColor);
            return refImg.mul(tf.scalar(1).sub(maskAlphaT)).add(colorMask.mul(maskAlphaT));
        });

        await tf.browser.toPixels(blendedPrompt, promptCanvas);

        // --- CLEANUP ---
        tf.dispose([promptTensor, queryTensor, prediction, blendedResult, blendedPrompt]);

    } catch (err) {
        console.error(err);
        alert("Inference Error! View Console.");
    }

    btnBtn.disabled = false;
    btnBtn.querySelector('.loader').style.display = 'none';
};

// Handle UI interactions
function setupUpload(inputId, imgId, dropId) {
    const input = document.getElementById(inputId);
    const dropArea = document.getElementById(dropId);

    input.onchange = (e) => {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = (f) => {
                document.getElementById(imgId).src = f.target.result;
                dropArea.classList.add('active-upload');
                dropArea.querySelector('span:nth-child(2)').innerText = "Loaded: " + e.target.files[0].name.substring(0, 12) + "...";
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    };
}

setupUpload('inputRef', 'imgRef', 'dropRef');
setupUpload('inputMask', 'imgMask', 'dropMask');
setupUpload('inputTarget', 'imgTarget', 'dropTarget');

init();
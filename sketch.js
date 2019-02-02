let looping = true;
let socket, cnvs, ctx, canvasDOM;
let fileName = "./frames/sketch";
let maxFrames = 20;

let m, b;

let x_vals = [];
let y_vals = [];

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    socket = io.connect('http://localhost:8080');
    cnvs = createCanvas(windowWidth, windowHeight);
    ctx = cnvs.drawingContext;
    canvasDOM = document.getElementById('defaultCanvas0');
    frameRate(30);
    background(200);
    fill(0);
    stroke(0);
    strokeWeight(1);
    if (!looping) {
        noLoop();
    }
    const mSeed = tf.scalar(random(1));
    const bSeed = tf.scalar(random(1));
    m = tf.variable(mSeed);
    b = tf.variable(bSeed);
    mSeed.dispose();
    bSeed.dispose();
}

function draw() {
    background(200);
    // optimizer.minimize(() => loss(f(xs), ys));
    if (x_vals.length >  0) {
        tf.tidy(() => {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
        });
        // optimizer.minimize(function() {
        //     loss(predict(xs), ys);
        // });
        noStroke();
        stroke(0);
        for (let i = 0; i < x_vals.length; i++) {
            let px = map(x_vals[i], 0, 1, 0, width);
            let py = map(y_vals[i], 0, 1, height, 0);
            ellipse(px, py, 5);
        }
        const lineX = [0, 1];
        const ys = tf.tidy(() => predict(lineX));
        const lineY = ys.dataSync();
        ys.dispose();
        let x1 = map(lineX[0], 0, 1, 0, width);
        let x2 = map(lineX[1], 0, 1, 0, width);
        let y1 = map(lineY[0], 0, 1, height, 0);
        let y2 = map(lineY[1], 0, 1, height, 0);
        line(x1, y1, x2, y2);
    }
    if (exporting && frameCount < maxFrames) {
        frameExport();
    }
}

function loss(pred, labels) {
    // (pred, label) => pred.sub(label).square().mean();
    return pred.sub(labels).square().mean();
}

function predict(x) {
    const xs = tf.tensor1d(x);
    // y = mx + b;
    const ys = xs.mul(m).add(b);
    return ys;
}

function mousePressed() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    x_vals.push(x);
    y_vals.push(y);
}

function keyPressed() {
    if (keyCode === 32) {
        if (looping) {
            noLoop();
            looping = false;
        } else {
            loop();
            looping = true;
        }
    }
    if (key == 'p' || key == 'P') {
        frameExport();
    }
    if (key == 'r' || key == 'R') {
        window.location.reload();
    }
    if (key == 'm' || key == 'M') {
        redraw();
    }
}
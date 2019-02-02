let looping = true;
let socket, cnvs, ctx, canvasDOM;
let fileName = "./frames/sketch";
let maxFrames = 20;

let a, b, c, d;

let x_vals = [];
let y_vals = [];

const learningRate = 0.5;
const optimizer = tf.train.adam(learningRate);

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
    const aSeed = tf.scalar(random(-1, 1));
    const bSeed = tf.scalar(random(-1, 1));
    const cSeed = tf.scalar(random(-1, 1));
    const dSeed = tf.scalar(random(-1, 1));
    a = tf.variable(aSeed);
    b = tf.variable(bSeed);
    c = tf.variable(cSeed);
    d = tf.variable(dSeed);
    aSeed.dispose();
    bSeed.dispose();
    cSeed.dispose();
    dSeed.dispose();
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
        fill(0);
        for (let i = 0; i < x_vals.length; i++) {
            let px = map(x_vals[i], -1, 1, 0, width);
            let py = map(y_vals[i], -1, 1, height, 0);
            ellipse(px, py, 5);
        }
        const curveX = [];
        for (let x = -1; x <= 1.05; x += 0.05) {
            curveX.push(x);
        }

        const ys = tf.tidy(() => predict(curveX));
        const curveY = ys.dataSync();
        noFill();
        ys.dispose();
        beginShape();
        for (let i = 0; i < curveX.length; i++) {
            let x = map(curveX[i], -1, 1, 0, width);
            let y = map(curveY[i], -1, 1, height, 0);
            vertex(x, y);
        }
        endShape();

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
    // const ys = xs.mul(m).add(b);

    // y = ax^2 + bx + c
    // const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
    const ys = xs.pow(tf.scalar(3)).mul(a)
        .add(xs.square().mul(b))
        .add(xs.mul(c))
        .add(d);
    return ys;
}

function mousePressed() {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
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
// const data = tf.tensor([1, 2, 3, 4]);
// const data = tf.tensor([0, 0, 127, 255, 1, 1, 2.2, 2, 10, 10, 20, 20], [3, 2, 2], "float32");

// first image 0, 0, 127, 255
// second image 1, 1, 2, 2
// i have two images, and they are of size 2 x 2


const values = [];
for (let i = 0; i < 9; i++) {
    values[i] = random(0, 100);
}
const shape = [3, 3];

const a = tf.tensor2d(values, shape, "int32");
const b = tf.tensor2d(values, shape, "int32");

// const c = a.add(b);
const c = a.matMul(b);


a.print();
// b.print();
c.print();

// const vtense = tf.variable(tense);
// console.log(tense);


// tense.print();
// console.log(data.toString());

// tense.data().then(function(stuff) {
//     console.log(stuff)
// });

// console.log(tense.dataSync());

// tf.tensor([1, 2, 3, 4]).print();
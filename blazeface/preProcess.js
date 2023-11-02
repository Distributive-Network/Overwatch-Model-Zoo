function preProcess(arrayBuffer, ort, inputNames){
	const jpeg = require("decodejpeg/decodeJPEG.js");
  const tf = require('tf.js');


  let imageData = jpeg.decode(arrayBuffer, { useTArray: true, formatAsRGBA: false });
  let width = imageData.width;
  let height= imageData.height;
  
  imageData = tf.tidy(()=>{
    let temp = tf.tensor( imageData.data, [height, width, 3], 'float32' );
    temp = tf.image.resizeBilinear( temp, [128,128] );
    temp = temp.div(255.).mul(2.).sub(1.);
    return temp.dataSync();
  });
  
  globalThis.imageWidth = width;
  globalThis.imageHeight = height;

  const x = new ort.Tensor('float32', Float32Array.from( imageData ), [1, 128, 128, 3]);
  const feeds = {
    'input:0': x
  };

  return feeds;
};

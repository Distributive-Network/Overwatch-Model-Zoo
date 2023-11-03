 function preprocess(val, ort, inputNames) { 
  const jpeg = require("decodejpeg/decodeJPEG.js");
  const tf = require('tf.js');
  let feeds = { };
  let dataArray = jpeg.decode( val, { useTArray: true , formatAsRGBA: false} );
  let height = dataArray.height;
  let width  = dataArray.width;

  dataArray = dataArray.data; // 0-255

  let tfTensor = tf.tensor( dataArray, [ 1, height, width, 3 ] );
  tfTensor = tf.image.resizeBilinear( tfTensor, [ 240, 320 ], true );
  tfTensor = tfTensor.sub(127).div(128);
  tfTensor = tf.transpose( tfTensor, [0, 3, 1, 2] );
  dataArray = Float32Array.from( tfTensor.dataSync() );

  const dataTensor = new ort.Tensor( 'float32',dataArray, [ 1, 3, 240, 320 ] );

  for ( i = 0; i < inputNames.length; i++ ) {
    feeds[ inputNames[i] ] = dataTensor;
  }

  return feeds 
}

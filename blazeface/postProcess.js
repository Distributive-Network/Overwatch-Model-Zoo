function postProcess( outputs, labels, outputNames){
  const tf = require('tf.js');

  /**
    * This function takes boxes as offsets from anchors
    * and produces the nonNormalized [ imageWidth - imageHeight ]
    * bounding boxes
    **/
  function decodeBoxes(boxes, anchors, imageSize){    
    const boxStarts = tf.slice(boxes,[0,0],[-1,2]);
    const centers = tf.add(boxStarts, anchors);
    const boxSizes = tf.slice(boxes, [0,2],[-1,2]);
    const boxSizesNormalized = tf.div(boxSizes, imageSize);
    const centersNormalized  = tf.div(centers, imageSize);
    const halfBoxSize = tf.div(boxSizesNormalized, 2);
    const starts = tf.sub(centersNormalized, halfBoxSize);
    const ends   = tf.add(centersNormalized, halfBoxSize);
    const startNormalized = tf.mul( starts, imageSize );
    const endsNormalized  = tf.mul( ends  , imageSize );
    return tf.concat2d([ startNormalized, endsNormalized ], 1);
  };
  
  /**
    *
    * A simple helper function which
    * takes an array of typedarrays
    * and generates a single typedarray that
    * is the concat of all those typedarrays
    *
    **/
  function mergeData( arr ){
    let totalLength = arr.reduce((acc, cur)=>{
      return acc + cur.data.length
    }, 0); 
    const retData = new Float32Array(totalLength);
    retData.fill(0, 0, totalLength);
    let index = 0;
    for (let arrInd = 0; arrInd < arr.length; arrInd++) {
      const curData = arr[arrInd].data;
      retData.set( curData, index );
      index += curData.length;
    };
    return retData;
  };
  
  /**
    * This function generates the anchors 
    * based on the blazeface defaults
    **/
  function generateAnchors(width, height){
    const anchors = [];
    let strides = [8, 16];
    let anchorsVals = [2, 6];
    for (let i = 0; i < 2; i++){
      const stride = strides[i];
      const gridRows = Math.floor((height + stride - 1) / stride);
      const gridCols = Math.floor((width + stride - 1) / stride);
      const anchorsNum = anchorsVals[i];
      
      for (let gridY = 0; gridY < gridRows; gridY++){
        const anchorY = stride * (gridY + 0.5);
        for (let gridX = 0; gridX < gridCols; gridX++){
          const anchorX = stride * (gridX + 0.5);
          for (let n = 0; n < anchorsNum; n++){
            anchors.push([anchorX, anchorY]);
          };
        };  
      };  
    };
    return tf.tensor2d(anchors);
  };



  let classes_output = [ outputs["Identity:0"], outputs["Identity_1:0"] ];
  let face_output = [ outputs["Identity_2:0"], outputs["Identity_3:0"] ];

  const out = tf.tidy(()=>{
    classes_output = mergeData( classes_output );
    face_output    = mergeData( face_output );

 
    let logits_tensor =  tf.expandDims(tf.tensor(classes_output), 1);
    let localization_tensor = tf.tensor(face_output, [ logits_tensor.shape[0], 16 ]);
    
    let anchors = generateAnchors(128, 128);
    
    let inputSize = tf.tensor([128, 128], [2]);
    let decodedBoxes = decodeBoxes( localization_tensor, anchors, inputSize );
    
    let scores_tensor = tf.squeeze(tf.sigmoid(logits_tensor));
    
    const boxIndices = tf.image.nonMaxSuppression(
      decodedBoxes, scores_tensor, 400, 0.5, 0.7
    ).arraySync();
    
    let boundingBoxes = tf.gather( decodedBoxes, boxIndices, axis = 0 );
    let scores = tf.gather( scores_tensor, boxIndices, axis = 0 );
    boundingBoxes = tf.div( boundingBoxes, [128, 128, 128, 128]);
    return { 'boxes': boundingBoxes.arraySync(), 'scores': scores.arraySync() };
  });
  
  return out;
};

function postProcess(out, labels, outputNames) {
  let boxOutput = [];
  let scoreOutput = [];

  boxOutput.push(out[outputNames[1]].data)
  scoreOutput.push(out[outputNames[0]].data)

  function nms(boxes, scores, conf, iou_threshold, numBoxes) {
    let bboxConf = [];
    let scoreConf = [];
    let newBbox = [];
    let newScore = [];
    let score;
    for (let ind=0; ind < numBoxes; ind+=1) {
      if (ind == 0) {
        score = scores.slice(ind, ind+2);
      } else {
        score = scores.slice((ind)*2, (ind + 1)*2);
      }
      if (score[1] > conf) {
        scoreConf.push(score);
        bboxConf.push( boxes.slice(ind*4, (ind+1)*4) );
      }
    };
    while(bboxConf.length > 0) {
      let current_box = bboxConf.shift();
      let current_score = scoreConf.shift();
      newBbox.push(current_box);
      newScore.push(current_score);
      let tempBoxes = [];
      let tempScores = [];

      let numGoodBoxes = bboxConf.length;
      for ( let ind=0; ind < numGoodBoxes; ind++ ) {
        let test_box = bboxConf[ind];
        let test_score = scoreConf[ind];

        const iou_value = iou( current_box.slice(0,4), test_box.slice(0,4) );
        if ( iou_value <= iou_threshold ) {
          tempBoxes.push( test_box );
          tempScores.push( test_score );
        };
      };
      bboxConf = tempBoxes;
      scoreConf = tempScores;
    };
    return { newBbox, newScore };
  };

  function iou(b1, b2) {
    let y11 = b1[1],
        x11 = b1[0],
        y12 = b1[3],
        x12 = b1[2];
    let y21 = b2[1],
        x21 = b2[0],
        y22 = b2[3],
        x22 = b2[2];
    let xI1 = Math.max( x11, x21 ),
        yI1 = Math.max( y11, y21 ),
        xI2 = Math.min( x12, x22 ),
        yI2 = Math.min( y12, y22 );
    let inter_area = Math.max( (xI2 - xI1), 0) * Math.max( (yI2 - yI1), 0);
    let b1_area = (x12 - x11) * (y12 - y11);
    let b2_area = (x22 - x21) * (y22 - y21);
    let union = (b1_area + b2_area) - inter_area;
    let eps = 0.0001;
    let iou = inter_area / (union + eps);
    return iou;
  };

  let numBoxes = (scoreOutput[0].length)/2;
  let numInputs = scoreOutput.length
  let finalBoxes = [];
  let finalScores = [];
  let threshold = 0.5;
  let iou_threshold = 0.5;
  for(let i = 0; i < numInputs; i += 1) {
    let { newBbox, newScore } = nms(boxOutput[i], scoreOutput[i], threshold, iou_threshold, numBoxes);
    finalBoxes.push(newBbox);
    finalScores.push(newScore);
  }
  
  return {finalBoxes, finalScores}
};

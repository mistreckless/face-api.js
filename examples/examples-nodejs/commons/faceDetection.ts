import * as faceapi from 'face-api.js';

export const faceDetectionNet = faceapi.nets.mtcnn
// export const faceDetectionNet = tinyFaceDetector

// SsdMobilenetv1Options
const minConfidence = 0.5

// TinyFaceDetectorOptions
const inputSize = 408
const scoreThreshold = 0.5

const minFaceSize = 40
const scaleFactor = 0.709
const maxNumScales = 10
const scoreThresholds =[0.6, 0.7, 0.7]
const  scaleSteps = undefined

function getFaceDetectorOptions(net: faceapi.NeuralNetwork<any>) {
  if (net === faceapi.nets.mtcnn) return new faceapi.MtcnnOptions({
    minFaceSize, scaleFactor, maxNumScales, scoreThresholds, scaleSteps
  })
  return net === faceapi.nets.ssdMobilenetv1
    ? new faceapi.SsdMobilenetv1Options({ minConfidence })
    : new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold })
}

export const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet)
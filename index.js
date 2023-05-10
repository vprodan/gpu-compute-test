import { GPU } from 'gpu.js';
import {
  createMatrixAddKernel,
  createMatrixMulKernel,
  gpuMatrixMulBlocks,
  makeBlocks,
  makeMatrix,
  measureTime
} from './tools.js'

const gpu = new GPU();

const matrixA = [
  [1, 2, 2, 2],
  [3, 1, 2, 2],
  [3, 3, 1, 2],
  [3, 3, 3, 1]
];
const matrixB = [
  [1, 2, 3, 4],
  [4, 2, 2, 1],
  [1, 1, 1, 1],
  [2, 1, 2, 1]
];

console.info('A:');
console.table(matrixA);

console.info('B:');
console.table(matrixB);

console.table(
  measureTime('A × B', () => createMatrixMulKernel(gpu, 4)(matrixA, matrixB))
);


const blockSize = 2;
const gpuAdd = createMatrixAddKernel(gpu, blockSize)
const gpuMul = createMatrixMulKernel(gpu, blockSize);

const aBlocks = makeBlocks(matrixA, blockSize);
const bBlocks = makeBlocks(matrixB, blockSize);

console.table(
  makeMatrix(
    measureTime(`A × B (blocks ${blockSize} × ${blockSize})`, () => gpuMatrixMulBlocks(gpuAdd, gpuMul, aBlocks, bBlocks, blockSize))
  )
);

await gpu.destroy();
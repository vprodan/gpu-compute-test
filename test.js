import { GPU } from 'gpu.js';
import {
  randomMatrix,
  makeBlocks,
  cpuMatrixMul,
  measureTime,
  createMatrixMulKernel,
  createMatrixAddKernel, gpuMatrixMulBlocks
} from './tools.js'

const n = 2000;
const matrixA = randomMatrix(n, n);
const matrixB = randomMatrix(n, n);

const gpu = new GPU();

const blockSize = n / 2;

const aBlocks = makeBlocks(matrixA, blockSize);
const bBlocks = makeBlocks(matrixB, blockSize);

const gpuAdd = createMatrixAddKernel(gpu, blockSize)
const gpuMul = createMatrixMulKernel(gpu, blockSize);

measureTime('GPU', () => gpuMatrixMulBlocks(gpuAdd, gpuMul, aBlocks, bBlocks));

await gpu.destroy();

measureTime('CPU', () => cpuMatrixMul(matrixA, matrixB));
/**
 * Generates random matrix
 * @param {number} x
 * @param {number} y
 */
export function randomMatrix(x, y) {
  const result = []

  for (let i = 0; i < x; i++) {
    const row = [];

    for (let j = 0; j < y; j++) {
      row.push(Math.floor(Math.random() * 10));
    }

    result.push(row);
  }

  return result;
}

/**
 * Generates empty matrix
 * @param {number} x
 * @param {number} y
 */
export function emptyMatrix(x, y) {
  const result = []

  for (let i = 0; i < x; i++) {
    result.push(new Array(y).fill(0));
  }

  return result;
}

/**
 * Divides up matrix into n blocks of `blockSize`
 * @param {number[][]} matrix 2D Matrix
 * @param {number} blockSize Block size
 * @return {number[][][][]} Matrix divided to Blocks
 */
export function makeBlocks(matrix, blockSize) {
  const numBlocks = Math.floor(matrix.length / blockSize);
  const blocks = [];

  for (let h = 0; h < numBlocks; h++) {
    const blockRow = [];

    for (let i = 0; i < numBlocks; i++) {
      const block = [];

      for (let j = h * blockSize; j < (h + 1) * blockSize; j++) {
        const row = [];

        for (let k = i * blockSize; k < (i + 1) * blockSize; k++) {
          row.push(matrix[j][k]);
        }

        block.push(row);
      }

      blockRow.push(block);
    }

    blocks.push(blockRow);
  }

  return blocks;
}

/**
 * Making a matrix from blocks
 * @param {number[][][][]} blocks Matrix divided to Blocks
 * @return {number[][]} Matrix
 */
export function makeMatrix(blocks) {
    const matrix = [];

    const blockRows = blocks.length;
    const blockCols = blocks[0].length;
    const blockSize = blocks[0][0].length;

    const matrixRows = blockRows * blockSize;
    const matrixCols = blockCols * blockSize;

    for (let i = 0; i < matrixRows; i++) {
      const row = [];

      for (let j = 0; j < matrixCols; j++) {
        const blockRow = Math.floor(i / blockSize);
        const blockCol = Math.floor(j / blockSize);
        const block = blocks[blockRow][blockCol];
        const cellRow = i % blockSize;
        const cellCol = j % blockSize;

        row.push(block[cellRow][cellCol]);
      }

      matrix.push(row);
    }

    return matrix;
}


/**
 * Multiplies matrix a to b using CPU
 * @param {number[][]} a 2D Matrix
 * @param {number[][]} b 2D Matrix
 */
export function cpuMatrixMul(a, b) {
  const result = [];

  for (let i = 0, len = a.length; i < len; i++) {
    const row = [];

    for (let j = 0, len = b[0].length; j < len; j++) {
      let sum = 0;

      for (let k = 0, len = a[0].length; k < len; k++) {
        sum += a[i][k] * b[k][j];
      }

      row.push(sum);
    }

    result.push(row);
  }

  return result;
}

/**
 * Measuring time of function execution
 * @param {string} label Label to measure
 * @param {Function} fn Function to measure
 * @return {any} Result of Function to measure
 */
export function measureTime(label, fn) {
  const start_time = performance.now();
  const output = fn();
  const stop_time = performance.now();
  console.log(`${label}: ${((stop_time - start_time) / 1000).toFixed(2)} s`);

  return output;
}

/**
 * Creates kernel for add matrix
 * @param {GPU} gpu GPU instance
 * @param {number} n Size of Matrix
 * @return {IKernelRunShortcut} Kernel Function
 */
export function createMatrixAddKernel(gpu, n) {
  return gpu.createKernel(function (a, b) {
    return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
  }, {
    output: [n, n],
    optimizeFloatMemory: true
  });
}

/**
 * Creates kernel for multiply matrix
 * @param {GPU} gpu GPU instance
 * @param {number} n Size of Matrix
 * @return {IKernelRunShortcut} Kernel Function
 */
export function createMatrixMulKernel(gpu, n) {
  return gpu.createKernel(function (a, b) {
    let sum = 0;

    for (let i = 0; i < this.constants.n; i++) {
      sum += a[this.thread.y][i] * b[i][this.thread.x];
    }

    return sum;
  }, {
    output: [n, n],
    constants: { n },
    optimizeFloatMemory: true
  });
}

/**
 * Multiplies blocks of matrix a to b
 * @param {Function} addFn Kernel Function for add
 * @param {Function} mulFn Kernel Function for multiply
 * @param {number[][][][]} a Matrix divided to blocks
 * @param {number[][][][]} b Matrix divided to blocks
 * @return {number[][][][]} Multiplied blocks
 */
export function gpuMatrixMulBlocks(addFn, mulFn, a, b) {
  const numBlocks = a.length;
  const x = a[0][0].length;
  const y = a[0][0][0].length;
  const EmptyMatrix = emptyMatrix(x, y);

  const output = [];

  for (let i = 0; i < numBlocks; i++) {
    const blockRow = [];

    for (let j = 0; j < numBlocks; j++) {
      let sum = [...EmptyMatrix];

      for (let k = 0; k < numBlocks; k++) {
        const result = mulFn(a[i][k], b[k][j]);
        sum = addFn(sum, result);
      }

      blockRow.push(sum.map(f32array => Array.from(f32array)));
    }

    output.push(blockRow);
  }

  return output;
}

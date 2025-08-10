const path = require('path');

module.exports = {
  entry: './src/userwhisperer.js',
  output: {
    path: path.resolve(__dirname, 'lib'),
    filename: 'userwhisperer.js',
    library: 'UserWhisperer',
    libraryTarget: 'umd',
    globalObject: 'this'
  },
  module: {
    rules: [
      {
        test: /\.(js|ts)$/,
        exclude: /node_modules/,
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true
          }
        }
      }
    ]
  },
  resolve: {
    extensions: ['.ts', '.js']
  },
  optimization: {
    minimize: true
  },
  externals: {
    // Add any external dependencies here if needed
  }
};

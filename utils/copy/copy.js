const glob = require('glob');
const path = require('path');
const fs = require('fs-extra');

console.log('copying')

const dest = '../../raw/voice'

glob('../../raw/TIMIT/**/*.WAV', function(err, files) {
  if (err) throw err;
  let parsed = [];
  files.forEach(file => {
    let relPath = path.relative('../../raw/TIMIT/', file)
    fs.copySync(file, path.join(dest, relPath.split(path.sep).join('_')))
  })
});
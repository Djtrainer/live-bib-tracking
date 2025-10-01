import pkg from './package.json' with { type: 'json' };
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Get all dependencies
const deps = Object.keys(pkg.dependencies);

// Read all source files and extract imports
function getAllImports(dir) {
  const imports = new Set();
  
  function readDir(currentDir) {
    const files = fs.readdirSync(currentDir);
    
    for (const file of files) {
      const filePath = path.join(currentDir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
        readDir(filePath);
      } else if (file.endsWith('.tsx') || file.endsWith('.ts')) {
        const content = fs.readFileSync(filePath, 'utf8');
        const importMatches = content.match(/^import.*from ['"]([^'"]+)['"]/gm);
        
        if (importMatches) {
          importMatches.forEach(match => {
            const importPath = match.match(/from ['"]([^'"]+)['"]/)[1];
            if (!importPath.startsWith('./') && !importPath.startsWith('../') && !importPath.startsWith('@/')) {
              imports.add(importPath);
            }
          });
        }
      }
    }
  }
  
  readDir(dir);
  return Array.from(imports);
}

const usedImports = getAllImports('./src');
console.log('Used imports:', usedImports);

// Find unused dependencies
const unused = deps.filter(dep => {
  // Check if dependency is used directly or as part of a scoped package
  return !usedImports.some(imp => imp === dep || imp.startsWith(dep + '/'));
});

console.log('\nUnused dependencies:');
unused.forEach(dep => console.log('- ' + dep));

console.log('\nPotential savings:');
console.log('Total dependencies:', deps.length);
console.log('Unused dependencies:', unused.length);
console.log('Percentage unused:', Math.round((unused.length / deps.length) * 100) + '%');

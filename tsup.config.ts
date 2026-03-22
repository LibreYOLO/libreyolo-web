import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  splitting: false,
  sourcemap: true,
  treeshake: true,
  external: ['onnxruntime-web'],
  esbuildOptions(options) {
    options.banner = {
      js: '/* libreyolo-web - MIT License - https://github.com/xuban-ceccon/libreyolo-web */',
    };
  },
});

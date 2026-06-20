// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Custom domain (gdl.fractalmanifold.com) is served at the root, so `base`
// stays "/". If you ever deploy to a project path, set `base` accordingly.
export default defineConfig({
  site: 'https://gdl.fractalmanifold.com',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});

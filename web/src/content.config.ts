import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Chapter content lives as Markdown in src/content/chapters/*.md
const chapters = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/chapters' }),
  schema: z.object({
    title: z.string(),
    order: z.number(),
    part: z.string().optional(),
    summary: z.string().optional(),
    draft: z.boolean().default(true),
  }),
});

export const collections = { chapters };

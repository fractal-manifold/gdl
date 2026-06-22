// Central place for everything language-dependent on the site.
// English is the default locale (served at "/"); Spanish lives under "/es/".

export const languages = { en: 'English', es: 'Español' } as const;
export type Lang = keyof typeof languages;
export const defaultLang: Lang = 'en';

/** Narrow an arbitrary string (e.g. a chapter id prefix or URL segment) to a Lang. */
export function isLang(value: string): value is Lang {
  return value === 'en' || value === 'es';
}

/** Derive the active language from a URL pathname ("/es/..." → "es"). */
export function getLangFromUrl(url: URL): Lang {
  const seg = url.pathname.split('/').filter(Boolean)[0];
  return seg && isLang(seg) ? seg : defaultLang;
}

/**
 * Prefix an absolute, English-based path with the locale segment.
 * `/` → `/es/`, `/chapters/` → `/es/chapters/`, `/#thesis` → `/es/#thesis`.
 */
export function localizePath(path: string, lang: Lang): string {
  if (lang === defaultLang) return path;
  if (path === '/') return `/${lang}/`;
  return `/${lang}${path}`;
}

/**
 * Map the current pathname onto its equivalent in `target` (used by the
 * language switcher). Strips any existing locale prefix first.
 */
export function switchLocalePath(pathname: string, target: Lang): string {
  const base = pathname.replace(/^\/(en|es)(?=\/|$)/, '') || '/';
  if (target === defaultLang) return base;
  return base === '/' ? `/${target}/` : `/${target}${base}`;
}

export const ui = {
  en: {
    'nav.thesis': 'The Thesis',
    'nav.gs': 'The 5 Gs',
    'nav.contents': 'Contents',
    'nav.getbook': 'Get the book',
    'footer.rights': 'All rights reserved.',
    'footer.top': 'Back to top',
    'chapters.title': 'Chapters',
    'chapters.read': 'Read the chapters online →',
    'lang.label': 'Language',
  },
  es: {
    'nav.thesis': 'La tesis',
    'nav.gs': 'Las 5 Ges',
    'nav.contents': 'Contenidos',
    'nav.getbook': 'Consigue el libro',
    'footer.rights': 'Todos los derechos reservados.',
    'footer.top': 'Volver arriba',
    'chapters.title': 'Capítulos',
    'chapters.read': 'Lee los capítulos en línea →',
    'lang.label': 'Idioma',
  },
} as const;

export function useTranslations(lang: Lang) {
  return function t(key: keyof (typeof ui)['en']): string {
    return ui[lang][key] ?? ui[defaultLang][key];
  };
}

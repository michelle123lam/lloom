import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "LLooM",
  description: "Concept Induction: Analyzing Unstructured Text with High-Level Concepts",
  base: "/lloom/",
  head: [
    ['link', { rel: 'icon', href: '/lloom/media/favicon.png' }],
    [
      'script',
      { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-YYD21V2BQE' }
    ],
    [
      'script',
      {},
      `window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-YYD21V2BQE');`
    ]
  ],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: {
      light: '/media/lloom.svg',
      dark: '/media/lloom.svg',
      alt: 'LLooM'
    },
    siteTitle: false,

    footer: {
      message: "LLooM is a research prototype produced by the <a href='https://hci.stanford.edu/'>Stanford Human-Computer Interaction Group</a> in collaboration with the <a href='https://idl.cs.washington.edu/'>UW Interactive Data Lab</a>."
    },

    search: {
      provider: "local"
    },

    nav: [
      { text: 'About', link: '/about/' },
      { text: 'API Reference', link: '/api/workbench' }
    ],

    sidebar: {
      '/': [
        {
          text: 'LLooM Overview',
          items: [
            { text: 'What is LLooM?', link: '/about/' },
            { text: 'Get Started', link: '/about/get-started' },
            { text: 'Using the LLooM Workbench', link: '/about/vis-guide' },
          ]
        },
        {
          text: 'Examples',
          items: [
            { text: 'Political Social Media', link: '/examples/pol-soc-media' },
            { text: 'Content Moderation', link: '/examples/content-mod' },
            { text: 'Academic Paper Abstracts', link: '/examples/paper-abstracts' },
            { text: 'AI Broader Impact Statements', link: '/examples/ai-impact-statements' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'LLooM Workbench', link: '/api/workbench' },
            { text: 'LLooM Operators', link: '/api/operators' },
          ]
        }
      ]
    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/michelle123lam/lloom' }
    ]
  }
})

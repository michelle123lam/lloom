# Vega Themes

[![npm version](https://img.shields.io/npm/v/vega-themes.svg)](https://www.npmjs.com/package/vega-themes)
[![Build Status](https://github.com/vega/vega-themes/workflows/Test/badge.svg)](https://github.com/vega/vega-themes/actions)
[![](https://data.jsdelivr.com/v1/package/npm/vega-themes/badge?style=rounded)](https://www.jsdelivr.com/package/npm/vega-themes)

Themes for stylized [Vega](https://vega.github.io/vega/) and [Vega-Lite](https://vega.github.io/vega-lite/) visualizations. For Vega version 3+ and Vega-Lite 2+.

A Vega *theme* is a [configuration object](https://vega.github.io/vega/docs/config/)
with default settings for a variety of visual properties such as colors, typefaces,
line widths and spacing. This module exports a set of named themes, which can be
passed as input to the Vega or Vega-Lite with [Vega-Embed](https://github.com/vega/vega-embed)
or directly as a configuration object to the [Vega parser](https://vega.github.io/vega/docs/api/parser/).

Try Vega-Themes in our [Observable Demo](https://beta.observablehq.com/@domoritz/vega-themes-demo).

Vega Themes comes with Vega-Embed:

```js
vegaEmbed(el, spec, {theme: 'quartz'});
```

If you want to use a different version, pass the theme as a configuration:

```js
vegaEmbed(el, spec, {config: vegaThemes.quartz});
```

Once instantiated, a visualization theme can not be changed. Instead, the
input specification must be re-parsed with a new theme.

## Included Themes

<a name="excel" href="#excel">#</a>
vega.themes.<b>excel</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-excel.ts "Source")

Chart theme modeled after Microsoft Excel. [Try it here](https://vega.github.io/vega-themes/?theme=excel).

<a name="ggplot2" href="#ggplot2">#</a>
vega.themes.<b>ggplot2</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-ggplot2.ts "Source")

Chart theme modeled after ggplot2. [Try it here](https://vega.github.io/vega-themes/?theme=ggplot2).

<a name="quartz" href="#quartz">#</a>
vega.themes.<b>quartz</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-quartz.ts "Source")

Chart theme modeled after Quartz. [Try it here](https://vega.github.io/vega-themes/?theme=quartz).

<a name="vox" href="#vox">#</a>
vega.themes.<b>vox</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-vox.ts "Source")

Chart theme modeled after Vox. [Try it here](https://vega.github.io/vega-themes/?theme=vox).

<a name="fivethirtyeight" href="#fivethirtyeight">#</a>
vega.themes.<b>fivethirtyeight</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-fivethirtyeight.ts "Source")

Chart theme modeled after FiveThirtyEight. [Try it here](https://vega.github.io/vega-themes/?theme=fivethirtyeight).

<a name="dark" href="#dark">#</a>
vega.themes.<b>dark</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-dark.ts "Source")

A dark theme. [Try it here](https://vega.github.io/vega-themes/?theme=dark).

<a name="latimes" href="#latimes">#</a>
vega.themes.<b>latimes</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-latimes.ts "Source")

Chart theme modeled after the Los Angeles Times. [Try it here](https://vega.github.io/vega-themes/?theme=latimes).

<a name="urbaninstitute" href="#urbaninstitute">#</a>
vega.themes.<b>urbaninstitute</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-urbaninstitute.ts "Source")

Chart theme modeled after the Urban Institute. [Try it here](https://vega.github.io/vega-themes/?theme=urbaninstitute).

<a name="googlecharts " href="#googlecharts">#</a>
vega.themes.<b>googlecharts</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-googlecharts.ts "Source")

Chart theme modeled after Google Charts. [Try it here](https://vega.github.io/vega-themes/?theme=googlecharts).

<a name="powerbi " href="#powerbi">#</a>
vega.themes.<b>powerbi</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-powerbi.ts "Source")

Chart theme modeled after Power BI Desktop default theme. [Try it here](https://vega.github.io/vega-themes/?theme=powerbi).

<a name="carbonwhite " href="#carbonwhite">#</a>
vega.themes.<b>carbonwhite</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-carbonwhite.ts "Source")

Chart theme modeled after IBM Carbon Charts - white theme. [Try it here](https://vega.github.io/vega-themes/?theme=carbonwhite).

<a name="carbong10" href="#carbong10">#</a>
vega.themes.<b>carbong10</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-carbong10.ts "Source")

Chart theme modeled after IBM Carbon Charts - grey 10 theme. This is the white theme with a slightly darker background. [Try it here](https://vega.github.io/vega-themes/?theme=carbong10).

<a name="carbong90" href="#carbong90">#</a>
vega.themes.<b>carbong90</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-carbong90.ts "Source")

Chart theme modeled after IBM Carbon Charts - grey 90 theme. [Try it here](https://vega.github.io/vega-themes/?theme=carbong90).

<a name="carbong100" href="#carbong100">#</a>
vega.themes.<b>carbong100</b>
[<>](https://github.com/vega/vega-themes/blob/main/src/theme-carbong100.ts "Source")

Chart theme modeled after IBM Carbon Charts - grey 100 theme. This is the grey 90 theme with a slightly darker background. [Try it here](https://vega.github.io/vega-themes/?theme=carbong100).


## Instructions for Developers

To view and test different themes, follow these steps:

1. Install dependencies via `yarn`.
2. Launch a local web server in the top-level directory with `yarn start`.
3. Make changes. The website will automatically reload.

## Publishing

To make a release, run `npm run release`.

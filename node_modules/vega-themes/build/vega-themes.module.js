var name = "vega-themes";
var version$1 = "2.14.0";
var description = "Themes for stylized Vega and Vega-Lite visualizations.";
var keywords = ["vega", "vega-lite", "themes", "style"];
var license = "BSD-3-Clause";
var author = {
  name: "UW Interactive Data Lab",
  url: "https://idl.cs.washington.edu"
};
var contributors = [{
  name: "Emily Gu",
  url: "https://github.com/emilygu"
}, {
  name: "Arvind Satyanarayan",
  url: "http://arvindsatya.com"
}, {
  name: "Jeffrey Heer",
  url: "https://idl.cs.washington.edu"
}, {
  name: "Dominik Moritz",
  url: "https://www.domoritz.de"
}];
var main = "build/vega-themes.js";
var module = "build/vega-themes.module.js";
var unpkg = "build/vega-themes.min.js";
var jsdelivr = "build/vega-themes.min.js";
var types = "build/vega-themes.module.d.ts";
var repository = {
  type: "git",
  url: "https://github.com/vega/vega-themes.git"
};
var files = ["src", "build"];
var scripts = {
  prebuild: "yarn clean",
  build: "rollup -c",
  clean: "rimraf build && rimraf examples/build",
  "copy:data": "rsync -r node_modules/vega-datasets/data/* examples/data",
  "copy:build": "rsync -r build/* examples/build",
  "deploy:gh": "yarn build && mkdir -p examples/build && rsync -r build/* examples/build && gh-pages -d examples",
  preversion: "yarn lint",
  serve: "browser-sync start -s -f build examples --serveStatic examples",
  start: "yarn build && concurrently --kill-others -n Server,Rollup 'yarn serve' 'rollup -c -w'",
  format: "eslint . --fix",
  lint: "eslint .",
  release: "release-it"
};
var devDependencies = {
  "@babel/core": "^7.22.9",
  "@babel/plugin-proposal-async-generator-functions": "^7.20.7",
  "@babel/plugin-proposal-json-strings": "^7.18.6",
  "@babel/plugin-proposal-object-rest-spread": "^7.20.7",
  "@babel/plugin-proposal-optional-catch-binding": "^7.18.6",
  "@babel/plugin-transform-runtime": "^7.22.9",
  "@babel/preset-env": "^7.22.9",
  "@babel/preset-typescript": "^7.22.5",
  "@release-it/conventional-changelog": "^7.0.0",
  "@rollup/plugin-json": "^6.0.0",
  "@rollup/plugin-node-resolve": "^15.1.0",
  "@rollup/plugin-terser": "^0.4.3",
  "@typescript-eslint/eslint-plugin": "^6.0.0",
  "@typescript-eslint/parser": "^6.0.0",
  "browser-sync": "^2.29.3",
  concurrently: "^8.2.0",
  eslint: "^8.45.0",
  "eslint-config-prettier": "^8.8.0",
  "eslint-plugin-prettier": "^5.0.0",
  "gh-pages": "^5.0.0",
  prettier: "^3.0.0",
  "release-it": "^16.1.0",
  rollup: "^3.26.2",
  "rollup-plugin-bundle-size": "^1.0.3",
  "rollup-plugin-ts": "^3.2.0",
  typescript: "^5.1.6",
  vega: "^5.25.0",
  "vega-lite": "^5.9.3"
};
var peerDependencies = {
  vega: "*",
  "vega-lite": "*"
};
var dependencies = {};
var pkg = {
  name: name,
  version: version$1,
  description: description,
  keywords: keywords,
  license: license,
  author: author,
  contributors: contributors,
  main: main,
  module: module,
  unpkg: unpkg,
  jsdelivr: jsdelivr,
  types: types,
  repository: repository,
  files: files,
  scripts: scripts,
  devDependencies: devDependencies,
  peerDependencies: peerDependencies,
  dependencies: dependencies
};

const lightColor = '#fff';
const medColor = '#888';
const darkTheme = {
  background: '#333',
  view: {
    stroke: medColor
  },
  title: {
    color: lightColor,
    subtitleColor: lightColor
  },
  style: {
    'guide-label': {
      fill: lightColor
    },
    'guide-title': {
      fill: lightColor
    }
  },
  axis: {
    domainColor: lightColor,
    gridColor: medColor,
    tickColor: lightColor
  }
};

const markColor$7 = '#4572a7';
const excelTheme = {
  background: '#fff',
  arc: {
    fill: markColor$7
  },
  area: {
    fill: markColor$7
  },
  line: {
    stroke: markColor$7,
    strokeWidth: 2
  },
  path: {
    stroke: markColor$7
  },
  rect: {
    fill: markColor$7
  },
  shape: {
    stroke: markColor$7
  },
  symbol: {
    fill: markColor$7,
    strokeWidth: 1.5,
    size: 50
  },
  axis: {
    bandPosition: 0.5,
    grid: true,
    gridColor: '#000000',
    gridOpacity: 1,
    gridWidth: 0.5,
    labelPadding: 10,
    tickSize: 5,
    tickWidth: 0.5
  },
  axisBand: {
    grid: false,
    tickExtra: true
  },
  legend: {
    labelBaseline: 'middle',
    labelFontSize: 11,
    symbolSize: 50,
    symbolType: 'square'
  },
  range: {
    category: ['#4572a7', '#aa4643', '#8aa453', '#71598e', '#4598ae', '#d98445', '#94aace', '#d09393', '#b9cc98', '#a99cbc']
  }
};

const markColor$6 = '#30a2da';
const axisColor$2 = '#cbcbcb';
const guideLabelColor = '#999';
const guideTitleColor = '#333';
const backgroundColor$2 = '#f0f0f0';
const blackTitle = '#333';
const fiveThirtyEightTheme = {
  arc: {
    fill: markColor$6
  },
  area: {
    fill: markColor$6
  },
  axis: {
    domainColor: axisColor$2,
    grid: true,
    gridColor: axisColor$2,
    gridWidth: 1,
    labelColor: guideLabelColor,
    labelFontSize: 10,
    titleColor: guideTitleColor,
    tickColor: axisColor$2,
    tickSize: 10,
    titleFontSize: 14,
    titlePadding: 10,
    labelPadding: 4
  },
  axisBand: {
    grid: false
  },
  background: backgroundColor$2,
  group: {
    fill: backgroundColor$2
  },
  legend: {
    labelColor: blackTitle,
    labelFontSize: 11,
    padding: 1,
    symbolSize: 30,
    symbolType: 'square',
    titleColor: blackTitle,
    titleFontSize: 14,
    titlePadding: 10
  },
  line: {
    stroke: markColor$6,
    strokeWidth: 2
  },
  path: {
    stroke: markColor$6,
    strokeWidth: 0.5
  },
  rect: {
    fill: markColor$6
  },
  range: {
    category: ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#b96db8', '#ff9e27', '#56cc60', '#52d2ca', '#52689e', '#545454', '#9fe4f8'],
    diverging: ['#cc0020', '#e77866', '#f6e7e1', '#d6e8ed', '#91bfd9', '#1d78b5'],
    heatmap: ['#d6e8ed', '#cee0e5', '#91bfd9', '#549cc6', '#1d78b5']
  },
  point: {
    filled: true,
    shape: 'circle'
  },
  shape: {
    stroke: markColor$6
  },
  bar: {
    binSpacing: 2,
    fill: markColor$6,
    stroke: null
  },
  title: {
    anchor: 'start',
    fontSize: 24,
    fontWeight: 600,
    offset: 20
  }
};

const markColor$5 = '#000';
const ggplot2Theme = {
  group: {
    fill: '#e5e5e5'
  },
  arc: {
    fill: markColor$5
  },
  area: {
    fill: markColor$5
  },
  line: {
    stroke: markColor$5
  },
  path: {
    stroke: markColor$5
  },
  rect: {
    fill: markColor$5
  },
  shape: {
    stroke: markColor$5
  },
  symbol: {
    fill: markColor$5,
    size: 40
  },
  axis: {
    domain: false,
    grid: true,
    gridColor: '#FFFFFF',
    gridOpacity: 1,
    labelColor: '#7F7F7F',
    labelPadding: 4,
    tickColor: '#7F7F7F',
    tickSize: 5.67,
    titleFontSize: 16,
    titleFontWeight: 'normal'
  },
  legend: {
    labelBaseline: 'middle',
    labelFontSize: 11,
    symbolSize: 40
  },
  range: {
    category: ['#000000', '#7F7F7F', '#1A1A1A', '#999999', '#333333', '#B0B0B0', '#4D4D4D', '#C9C9C9', '#666666', '#DCDCDC']
  }
};

const headlineFontSize = 22;
const headlineFontWeight = 'normal';
const labelFont$1 = 'Benton Gothic, sans-serif';
const labelFontSize = 11.5;
const labelFontWeight = 'normal';
const markColor$4 = '#82c6df';
// const markHighlight = '#006d8f';
// const markDemocrat = '#5789b8';
// const markRepublican = '#d94f54';
const titleFont = 'Benton Gothic Bold, sans-serif';
const titleFontWeight = 'normal';
const titleFontSize$1 = 13;
const colorSchemes$1 = {
  'category-6': ['#ec8431', '#829eb1', '#c89d29', '#3580b1', '#adc839', '#ab7fb4'],
  'fire-7': ['#fbf2c7', '#f9e39c', '#f8d36e', '#f4bb6a', '#e68a4f', '#d15a40', '#ab4232'],
  'fireandice-6': ['#e68a4f', '#f4bb6a', '#f9e39c', '#dadfe2', '#a6b7c6', '#849eae'],
  'ice-7': ['#edefee', '#dadfe2', '#c4ccd2', '#a6b7c6', '#849eae', '#607785', '#47525d']
};
const latimesTheme = {
  background: '#ffffff',
  title: {
    anchor: 'start',
    color: '#000000',
    font: titleFont,
    fontSize: headlineFontSize,
    fontWeight: headlineFontWeight
  },
  arc: {
    fill: markColor$4
  },
  area: {
    fill: markColor$4
  },
  line: {
    stroke: markColor$4,
    strokeWidth: 2
  },
  path: {
    stroke: markColor$4
  },
  rect: {
    fill: markColor$4
  },
  shape: {
    stroke: markColor$4
  },
  symbol: {
    fill: markColor$4,
    size: 30
  },
  axis: {
    labelFont: labelFont$1,
    labelFontSize,
    labelFontWeight,
    titleFont,
    titleFontSize: titleFontSize$1,
    titleFontWeight
  },
  axisX: {
    labelAngle: 0,
    labelPadding: 4,
    tickSize: 3
  },
  axisY: {
    labelBaseline: 'middle',
    maxExtent: 45,
    minExtent: 45,
    tickSize: 2,
    titleAlign: 'left',
    titleAngle: 0,
    titleX: -45,
    titleY: -11
  },
  legend: {
    labelFont: labelFont$1,
    labelFontSize,
    symbolType: 'square',
    titleFont,
    titleFontSize: titleFontSize$1,
    titleFontWeight
  },
  range: {
    category: colorSchemes$1['category-6'],
    diverging: colorSchemes$1['fireandice-6'],
    heatmap: colorSchemes$1['fire-7'],
    ordinal: colorSchemes$1['fire-7'],
    ramp: colorSchemes$1['fire-7']
  }
};

const markColor$3 = '#ab5787';
const axisColor$1 = '#979797';
const quartzTheme = {
  background: '#f9f9f9',
  arc: {
    fill: markColor$3
  },
  area: {
    fill: markColor$3
  },
  line: {
    stroke: markColor$3
  },
  path: {
    stroke: markColor$3
  },
  rect: {
    fill: markColor$3
  },
  shape: {
    stroke: markColor$3
  },
  symbol: {
    fill: markColor$3,
    size: 30
  },
  axis: {
    domainColor: axisColor$1,
    domainWidth: 0.5,
    gridWidth: 0.2,
    labelColor: axisColor$1,
    tickColor: axisColor$1,
    tickWidth: 0.2,
    titleColor: axisColor$1
  },
  axisBand: {
    grid: false
  },
  axisX: {
    grid: true,
    tickSize: 10
  },
  axisY: {
    domain: false,
    grid: true,
    tickSize: 0
  },
  legend: {
    labelFontSize: 11,
    padding: 1,
    symbolSize: 30,
    symbolType: 'square'
  },
  range: {
    category: ['#ab5787', '#51b2e5', '#703c5c', '#168dd9', '#d190b6', '#00609f', '#d365ba', '#154866', '#666666', '#c4c4c4']
  }
};

const markColor$2 = '#3e5c69';
const voxTheme = {
  background: '#fff',
  arc: {
    fill: markColor$2
  },
  area: {
    fill: markColor$2
  },
  line: {
    stroke: markColor$2
  },
  path: {
    stroke: markColor$2
  },
  rect: {
    fill: markColor$2
  },
  shape: {
    stroke: markColor$2
  },
  symbol: {
    fill: markColor$2
  },
  axis: {
    domainWidth: 0.5,
    grid: true,
    labelPadding: 2,
    tickSize: 5,
    tickWidth: 0.5,
    titleFontWeight: 'normal'
  },
  axisBand: {
    grid: false
  },
  axisX: {
    gridWidth: 0.2
  },
  axisY: {
    gridDash: [3],
    gridWidth: 0.4
  },
  legend: {
    labelFontSize: 11,
    padding: 1,
    symbolType: 'square'
  },
  range: {
    category: ['#3e5c69', '#6793a6', '#182429', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#e2ddf2']
  }
};

const markColor$1 = '#1696d2';
const axisColor = '#000000';
const backgroundColor$1 = '#FFFFFF';
const font = 'Lato';
const labelFont = 'Lato';
const sourceFont = 'Lato';
const gridColor$1 = '#DEDDDD';
const titleFontSize = 18;
const colorSchemes = {
  'main-colors': ['#1696d2', '#d2d2d2', '#000000', '#fdbf11', '#ec008b', '#55b748', '#5c5859', '#db2b27'],
  'shades-blue': ['#CFE8F3', '#A2D4EC', '#73BFE2', '#46ABDB', '#1696D2', '#12719E', '#0A4C6A', '#062635'],
  'shades-gray': ['#F5F5F5', '#ECECEC', '#E3E3E3', '#DCDBDB', '#D2D2D2', '#9D9D9D', '#696969', '#353535'],
  'shades-yellow': ['#FFF2CF', '#FCE39E', '#FDD870', '#FCCB41', '#FDBF11', '#E88E2D', '#CA5800', '#843215'],
  'shades-magenta': ['#F5CBDF', '#EB99C2', '#E46AA7', '#E54096', '#EC008B', '#AF1F6B', '#761548', '#351123'],
  'shades-green': ['#DCEDD9', '#BCDEB4', '#98CF90', '#78C26D', '#55B748', '#408941', '#2C5C2D', '#1A2E19'],
  'shades-black': ['#D5D5D4', '#ADABAC', '#848081', '#5C5859', '#332D2F', '#262223', '#1A1717', '#0E0C0D'],
  'shades-red': ['#F8D5D4', '#F1AAA9', '#E9807D', '#E25552', '#DB2B27', '#A4201D', '#6E1614', '#370B0A'],
  'one-group': ['#1696d2', '#000000'],
  'two-groups-cat-1': ['#1696d2', '#000000'],
  'two-groups-cat-2': ['#1696d2', '#fdbf11'],
  'two-groups-cat-3': ['#1696d2', '#db2b27'],
  'two-groups-seq': ['#a2d4ec', '#1696d2'],
  'three-groups-cat': ['#1696d2', '#fdbf11', '#000000'],
  'three-groups-seq': ['#a2d4ec', '#1696d2', '#0a4c6a'],
  'four-groups-cat-1': ['#000000', '#d2d2d2', '#fdbf11', '#1696d2'],
  'four-groups-cat-2': ['#1696d2', '#ec0008b', '#fdbf11', '#5c5859'],
  'four-groups-seq': ['#cfe8f3', '#73bf42', '#1696d2', '#0a4c6a'],
  'five-groups-cat-1': ['#1696d2', '#fdbf11', '#d2d2d2', '#ec008b', '#000000'],
  'five-groups-cat-2': ['#1696d2', '#0a4c6a', '#d2d2d2', '#fdbf11', '#332d2f'],
  'five-groups-seq': ['#cfe8f3', '#73bf42', '#1696d2', '#0a4c6a', '#000000'],
  'six-groups-cat-1': ['#1696d2', '#ec008b', '#fdbf11', '#000000', '#d2d2d2', '#55b748'],
  'six-groups-cat-2': ['#1696d2', '#d2d2d2', '#ec008b', '#fdbf11', '#332d2f', '#0a4c6a'],
  'six-groups-seq': ['#cfe8f3', '#a2d4ec', '#73bfe2', '#46abdb', '#1696d2', '#12719e'],
  'diverging-colors': ['#ca5800', '#fdbf11', '#fdd870', '#fff2cf', '#cfe8f3', '#73bfe2', '#1696d2', '#0a4c6a']
};
const urbanInstituteTheme = {
  background: backgroundColor$1,
  title: {
    anchor: 'start',
    fontSize: titleFontSize,
    font: font
  },
  axisX: {
    domain: true,
    domainColor: axisColor,
    domainWidth: 1,
    grid: false,
    labelFontSize: 12,
    labelFont: labelFont,
    labelAngle: 0,
    tickColor: axisColor,
    tickSize: 5,
    titleFontSize: 12,
    titlePadding: 10,
    titleFont: font
  },
  axisY: {
    domain: false,
    domainWidth: 1,
    grid: true,
    gridColor: gridColor$1,
    gridWidth: 1,
    labelFontSize: 12,
    labelFont: labelFont,
    labelPadding: 8,
    ticks: false,
    titleFontSize: 12,
    titlePadding: 10,
    titleFont: font,
    titleAngle: 0,
    titleY: -10,
    titleX: 18
  },
  legend: {
    labelFontSize: 12,
    labelFont: labelFont,
    symbolSize: 100,
    titleFontSize: 12,
    titlePadding: 10,
    titleFont: font,
    orient: 'right',
    offset: 10
  },
  view: {
    stroke: 'transparent'
  },
  range: {
    category: colorSchemes['six-groups-cat-1'],
    diverging: colorSchemes['diverging-colors'],
    heatmap: colorSchemes['diverging-colors'],
    ordinal: colorSchemes['six-groups-seq'],
    ramp: colorSchemes['shades-blue']
  },
  area: {
    fill: markColor$1
  },
  rect: {
    fill: markColor$1
  },
  line: {
    color: markColor$1,
    stroke: markColor$1,
    strokeWidth: 5
  },
  trail: {
    color: markColor$1,
    stroke: markColor$1,
    strokeWidth: 0,
    size: 1
  },
  path: {
    stroke: markColor$1,
    strokeWidth: 0.5
  },
  point: {
    filled: true
  },
  text: {
    font: sourceFont,
    color: markColor$1,
    fontSize: 11,
    align: 'center',
    fontWeight: 400,
    size: 11
  },
  style: {
    bar: {
      fill: markColor$1,
      stroke: null
    }
  },
  arc: {
    fill: markColor$1
  },
  shape: {
    stroke: markColor$1
  },
  symbol: {
    fill: markColor$1,
    size: 30
  }
};

/**
 * Copyright 2020 Google LLC.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

const markColor = '#3366CC';
const gridColor = '#ccc';
const defaultFont$1 = 'Arial, sans-serif';
const googlechartsTheme = {
  arc: {
    fill: markColor
  },
  area: {
    fill: markColor
  },
  path: {
    stroke: markColor
  },
  rect: {
    fill: markColor
  },
  shape: {
    stroke: markColor
  },
  symbol: {
    stroke: markColor
  },
  circle: {
    fill: markColor
  },
  background: '#fff',
  padding: {
    top: 10,
    right: 10,
    bottom: 10,
    left: 10
  },
  style: {
    'guide-label': {
      font: defaultFont$1,
      fontSize: 12
    },
    'guide-title': {
      font: defaultFont$1,
      fontSize: 12
    },
    'group-title': {
      font: defaultFont$1,
      fontSize: 12
    }
  },
  title: {
    font: defaultFont$1,
    fontSize: 14,
    fontWeight: 'bold',
    dy: -3,
    anchor: 'start'
  },
  axis: {
    gridColor: gridColor,
    tickColor: gridColor,
    domain: false,
    grid: true
  },
  range: {
    category: ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#AB47BC', '#00ACC1', '#FF7043', '#9E9D24', '#5C6BC0', '#F06292', '#00796B', '#C2185B'],
    heatmap: ['#c6dafc', '#5e97f6', '#2a56c6']
  }
};

const ptToPx = value => value * (1 / 3 + 1);
const fontSmallPx = ptToPx(9);
const legendFontPx = ptToPx(10);
const fontLargePx = ptToPx(12);
const fontStandard = 'Segoe UI';
const fontTitle = 'wf_standard-font, helvetica, arial, sans-serif';
const firstLevelElementColor = '#252423';
const secondLevelElementColor = '#605E5C';
const backgroundColor = 'transparent';
const backgroundSecondaryColor = '#C8C6C4';
const paletteColor1 = '#118DFF';
const paletteColor2 = '#12239E';
const paletteColor3 = '#E66C37';
const paletteColor4 = '#6B007B';
const paletteColor5 = '#E044A7';
const paletteColor6 = '#744EC2';
const paletteColor7 = '#D9B300';
const paletteColor8 = '#D64550';
const divergentColorMax = paletteColor1;
const divergentColorMin = '#DEEFFF';
const divergentPalette = [divergentColorMin, divergentColorMax];
const ordinalPalette = [divergentColorMin, '#c7e4ff', '#b0d9ff', '#9aceff', '#83c3ff', '#6cb9ff', '#55aeff', '#3fa3ff', '#2898ff', divergentColorMax];
const powerbiTheme = {
  view: {
    stroke: backgroundColor
  },
  background: backgroundColor,
  font: fontStandard,
  header: {
    titleFont: fontTitle,
    titleFontSize: fontLargePx,
    titleColor: firstLevelElementColor,
    labelFont: fontStandard,
    labelFontSize: legendFontPx,
    labelColor: secondLevelElementColor
  },
  axis: {
    ticks: false,
    grid: false,
    domain: false,
    labelColor: secondLevelElementColor,
    labelFontSize: fontSmallPx,
    titleFont: fontTitle,
    titleColor: firstLevelElementColor,
    titleFontSize: fontLargePx,
    titleFontWeight: 'normal'
  },
  axisQuantitative: {
    tickCount: 3,
    grid: true,
    gridColor: backgroundSecondaryColor,
    gridDash: [1, 5],
    labelFlush: false
  },
  axisBand: {
    tickExtra: true
  },
  axisX: {
    labelPadding: 5
  },
  axisY: {
    labelPadding: 10
  },
  bar: {
    fill: paletteColor1
  },
  line: {
    stroke: paletteColor1,
    strokeWidth: 3,
    strokeCap: 'round',
    strokeJoin: 'round'
  },
  text: {
    font: fontStandard,
    fontSize: fontSmallPx,
    fill: secondLevelElementColor
  },
  arc: {
    fill: paletteColor1
  },
  area: {
    fill: paletteColor1,
    line: true,
    opacity: 0.6
  },
  path: {
    stroke: paletteColor1
  },
  rect: {
    fill: paletteColor1
  },
  point: {
    fill: paletteColor1,
    filled: true,
    size: 75
  },
  shape: {
    stroke: paletteColor1
  },
  symbol: {
    fill: paletteColor1,
    strokeWidth: 1.5,
    size: 50
  },
  legend: {
    titleFont: fontStandard,
    titleFontWeight: 'bold',
    titleColor: secondLevelElementColor,
    labelFont: fontStandard,
    labelFontSize: legendFontPx,
    labelColor: secondLevelElementColor,
    symbolType: 'circle',
    symbolSize: 75
  },
  range: {
    category: [paletteColor1, paletteColor2, paletteColor3, paletteColor4, paletteColor5, paletteColor6, paletteColor7, paletteColor8],
    diverging: divergentPalette,
    heatmap: divergentPalette,
    ordinal: ordinalPalette
  }
};

const defaultFont = 'IBM Plex Sans,system-ui,-apple-system,BlinkMacSystemFont,".sfnstext-regular",sans-serif';
const fontWeight = 400;
const darkCategories = ['#8a3ffc', '#33b1ff', '#007d79', '#ff7eb6', '#fa4d56', '#fff1f1', '#6fdc8c', '#4589ff', '#d12771', '#d2a106', '#08bdba', '#bae6ff', '#ba4e00', '#d4bbff'];
const lightCategories = ['#6929c4', '#1192e8', '#005d5d', '#9f1853', '#fa4d56', '#570408', '#198038', '#002d9c', '#ee538b', '#b28600', '#009d9a', '#012749', '#8a3800', '#a56eff'];
function genCarbonConfig({
  type,
  background
}) {
  const viewbg = type === 'dark' ? '#161616' : '#ffffff';
  const textColor = type === 'dark' ? '#f4f4f4' : '#161616';
  const category = type === 'dark' ? darkCategories : lightCategories;
  const markColor = type === 'dark' ? '#d4bbff' : '#6929c4';
  return {
    background,
    arc: {
      fill: markColor
    },
    area: {
      fill: markColor
    },
    path: {
      stroke: markColor
    },
    rect: {
      fill: markColor
    },
    shape: {
      stroke: markColor
    },
    symbol: {
      stroke: markColor
    },
    circle: {
      fill: markColor
    },
    view: {
      fill: viewbg,
      stroke: viewbg
    },
    group: {
      fill: viewbg
    },
    title: {
      color: textColor,
      anchor: 'start',
      dy: -15,
      fontSize: 16,
      font: defaultFont,
      fontWeight: 600
    },
    axis: {
      labelColor: textColor,
      labelFontSize: 12,
      grid: true,
      gridColor: '#525252',
      titleColor: textColor,
      labelAngle: 0
    },
    style: {
      'guide-label': {
        font: defaultFont,
        fill: textColor,
        fontWeight: fontWeight
      },
      'guide-title': {
        font: defaultFont,
        fill: textColor,
        fontWeight: fontWeight
      }
    },
    range: {
      category,
      diverging: ['#750e13', '#a2191f', '#da1e28', '#fa4d56', '#ff8389', '#ffb3b8', '#ffd7d9', '#fff1f1', '#e5f6ff', '#bae6ff', '#82cfff', '#33b1ff', '#1192e8', '#0072c3', '#00539a', '#003a6d'],
      heatmap: ['#f6f2ff', '#e8daff', '#d4bbff', '#be95ff', '#a56eff', '#8a3ffc', '#6929c4', '#491d8b', '#31135e', '#1c0f30']
    }
  };
}

const carbonwhite = genCarbonConfig({
  type: 'light',
  background: '#ffffff'
});

const carbong10 = genCarbonConfig({
  type: 'light',
  background: '#f4f4f4'
});

const carbong90 = genCarbonConfig({
  type: 'dark',
  background: '#262626'
});

const carbong100 = genCarbonConfig({
  type: 'dark',
  background: '#161616'
});

const version = pkg.version;

export { carbong10, carbong100, carbong90, carbonwhite, darkTheme as dark, excelTheme as excel, fiveThirtyEightTheme as fivethirtyeight, ggplot2Theme as ggplot2, googlechartsTheme as googlecharts, latimesTheme as latimes, powerbiTheme as powerbi, quartzTheme as quartz, urbanInstituteTheme as urbaninstitute, version, voxTheme as vox };

import {Config} from './config';

const markColor = '#1696d2';
const axisColor = '#000000';
const backgroundColor = '#FFFFFF';
const font = 'Lato';
const labelFont = 'Lato';
const sourceFont = 'Lato';
const gridColor = '#DEDDDD';
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
  'diverging-colors': ['#ca5800', '#fdbf11', '#fdd870', '#fff2cf', '#cfe8f3', '#73bfe2', '#1696d2', '#0a4c6a'],
};

const urbanInstituteTheme: Config = {
  background: backgroundColor,

  title: {
    anchor: 'start',
    fontSize: titleFontSize,
    font: font,
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
    titleFont: font,
  },

  axisY: {
    domain: false,
    domainWidth: 1,
    grid: true,
    gridColor: gridColor,
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
    titleX: 18,
  },

  legend: {
    labelFontSize: 12,
    labelFont: labelFont,
    symbolSize: 100,
    titleFontSize: 12,
    titlePadding: 10,
    titleFont: font,
    orient: 'right',
    offset: 10,
  },

  view: {
    stroke: 'transparent',
  },

  range: {
    category: colorSchemes['six-groups-cat-1'],
    diverging: colorSchemes['diverging-colors'],
    heatmap: colorSchemes['diverging-colors'],
    ordinal: colorSchemes['six-groups-seq'],
    ramp: colorSchemes['shades-blue'],
  },

  area: {
    fill: markColor,
  },

  rect: {
    fill: markColor,
  },

  line: {
    color: markColor,
    stroke: markColor,
    strokeWidth: 5,
  },

  trail: {
    color: markColor,
    stroke: markColor,
    strokeWidth: 0,
    size: 1,
  },

  path: {
    stroke: markColor,
    strokeWidth: 0.5,
  },

  point: {
    filled: true,
  },

  text: {
    font: sourceFont,
    color: markColor,
    fontSize: 11,
    align: 'center',
    fontWeight: 400,
    size: 11,
  },

  style: {
    bar: {
      fill: markColor,
      stroke: null,
    },
  },

  arc: {fill: markColor},
  shape: {stroke: markColor},
  symbol: {fill: markColor, size: 30},
};

export default urbanInstituteTheme;

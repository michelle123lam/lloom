import {Config} from './config';

const defaultFont = 'IBM Plex Sans,system-ui,-apple-system,BlinkMacSystemFont,".sfnstext-regular",sans-serif';
const fontWeight = 400;

const darkCategories = [
  '#8a3ffc',
  '#33b1ff',
  '#007d79',
  '#ff7eb6',
  '#fa4d56',
  '#fff1f1',
  '#6fdc8c',
  '#4589ff',
  '#d12771',
  '#d2a106',
  '#08bdba',
  '#bae6ff',
  '#ba4e00',
  '#d4bbff',
];

const lightCategories = [
  '#6929c4',
  '#1192e8',
  '#005d5d',
  '#9f1853',
  '#fa4d56',
  '#570408',
  '#198038',
  '#002d9c',
  '#ee538b',
  '#b28600',
  '#009d9a',
  '#012749',
  '#8a3800',
  '#a56eff',
];

function genCarbonConfig({type, background}: {type: 'light' | 'dark'; background: string}): Config {
  const viewbg = type === 'dark' ? '#161616' : '#ffffff';
  const textColor = type === 'dark' ? '#f4f4f4' : '#161616';
  const category = type === 'dark' ? darkCategories : lightCategories;
  const markColor = type === 'dark' ? '#d4bbff' : '#6929c4';

  return {
    background,

    arc: {fill: markColor},
    area: {fill: markColor},
    path: {stroke: markColor},
    rect: {fill: markColor},
    shape: {stroke: markColor},
    symbol: {stroke: markColor},
    circle: {fill: markColor},

    view: {
      fill: viewbg,
      stroke: viewbg,
    },

    group: {
      fill: viewbg,
    },

    title: {
      color: textColor,
      anchor: 'start',
      dy: -15,
      fontSize: 16,
      font: defaultFont,
      fontWeight: 600,
    },

    axis: {
      labelColor: textColor,
      labelFontSize: 12,
      grid: true,
      gridColor: '#525252',
      titleColor: textColor,
      labelAngle: 0,
    },

    style: {
      'guide-label': {
        font: defaultFont,
        fill: textColor,
        fontWeight: fontWeight,
      },
      'guide-title': {
        font: defaultFont,
        fill: textColor,
        fontWeight: fontWeight,
      },
    },

    range: {
      category,
      diverging: [
        '#750e13',
        '#a2191f',
        '#da1e28',
        '#fa4d56',
        '#ff8389',
        '#ffb3b8',
        '#ffd7d9',
        '#fff1f1',
        '#e5f6ff',
        '#bae6ff',
        '#82cfff',
        '#33b1ff',
        '#1192e8',
        '#0072c3',
        '#00539a',
        '#003a6d',
      ],
      heatmap: [
        '#f6f2ff',
        '#e8daff',
        '#d4bbff',
        '#be95ff',
        '#a56eff',
        '#8a3ffc',
        '#6929c4',
        '#491d8b',
        '#31135e',
        '#1c0f30',
      ],
    },
  };
}

export default genCarbonConfig;

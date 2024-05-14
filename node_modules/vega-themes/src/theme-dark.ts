import {Config} from './config';

const lightColor = '#fff';
const medColor = '#888';

const darkTheme: Config = {
  background: '#333',

  view: {
    stroke: medColor,
  },

  title: {
    color: lightColor,
    subtitleColor: lightColor,
  },

  style: {
    'guide-label': {
      fill: lightColor,
    },
    'guide-title': {
      fill: lightColor,
    },
  },

  axis: {
    domainColor: lightColor,
    gridColor: medColor,
    tickColor: lightColor,
  },
};

export default darkTheme;

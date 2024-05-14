import {formatValue} from './formatValue';
import defaultStyle from './style';

const EL_ID = 'vg-tooltip-element';

export const DEFAULT_OPTIONS = {
  /**
   * X offset.
   */
  offsetX: 10,

  /**
   * Y offset.
   */
  offsetY: 10,

  /**
   * ID of the tooltip element.
   */
  id: EL_ID,

  /**
   * ID of the tooltip CSS style.
   */
  styleId: 'vega-tooltip-style',

  /**
   * The name of the theme. You can use the CSS class called [THEME]-theme to style the tooltips.
   *
   * There are two predefined themes: "light" (default) and "dark".
   */
  theme: 'light',

  /**
   * Do not use the default styles provided by Vega Tooltip. If you enable this option, you need to use your own styles. It is not necessary to disable the default style when using a custom theme.
   */
  disableDefaultStyle: false,

  /**
   * HTML sanitizer function that removes dangerous HTML to prevent XSS.
   *
   * This should be a function from string to string. You may replace it with a formatter such as a markdown formatter.
   */
  sanitize: escapeHTML,

  /**
   * The maximum recursion depth when printing objects in the tooltip.
   */
  maxDepth: 2,

  /**
   * A function to customize the rendered HTML of the tooltip.
   * @param value A value string, or object of value strings keyed by field
   * @param sanitize The `sanitize` function from `options.sanitize`
   * @param baseURL The `baseURL` from `options.baseURL`
   * @returns {string} The returned string will become the `innerHTML` of the tooltip element
   */
  formatTooltip: formatValue,

  /**
   * The baseurl to use in image paths.
   */
  baseURL: '',
};

export type Options = Partial<typeof DEFAULT_OPTIONS>;

/**
 * Escape special HTML characters.
 *
 * @param value A value to convert to string and HTML-escape.
 */
export function escapeHTML(value: any): string {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;');
}

export function createDefaultStyle(id: string): string {
  // Just in case this id comes from a user, ensure these is no security issues
  if (!/^[A-Za-z]+[-:.\w]*$/.test(id)) {
    throw new Error('Invalid HTML ID');
  }

  return defaultStyle.toString().replace(EL_ID, id);
}

import {TooltipHandler} from 'vega-typings';

import {createDefaultStyle, DEFAULT_OPTIONS, Options} from './defaults';
import {calculatePosition} from './position';

/**
 * The tooltip handler class.
 */
export class Handler {
  /**
   * The handler function. We bind this to this function in the constructor.
   */
  public call: TooltipHandler;

  /**
   * Complete tooltip options.
   */
  private options: Required<Options>;

  /**
   * The tooltip html element.
   */
  private el: HTMLElement | null;

  /**
   * Create the tooltip handler and initialize the element and style.
   *
   * @param options Tooltip Options
   */
  constructor(options?: Options) {
    this.options = {...DEFAULT_OPTIONS, ...options};
    const elementId = this.options.id;
    this.el = null;

    // bind this to call
    this.call = this.tooltipHandler.bind(this);

    // prepend a default stylesheet for tooltips to the head
    if (!this.options.disableDefaultStyle && !document.getElementById(this.options.styleId)) {
      const style = document.createElement('style');
      style.setAttribute('id', this.options.styleId);
      style.innerHTML = createDefaultStyle(elementId);

      const head = document.head;
      if (head.childNodes.length > 0) {
        head.insertBefore(style, head.childNodes[0]);
      } else {
        head.appendChild(style);
      }
    }
  }

  /**
   * The tooltip handler function.
   */
  private tooltipHandler(handler: any, event: MouseEvent, item: any, value: any) {
    // console.log(handler, event, item, value);

    // append a div element that we use as a tooltip unless it already exists
    this.el = document.getElementById(this.options.id);
    if (!this.el) {
      this.el = document.createElement('div');
      this.el.setAttribute('id', this.options.id);
      this.el.classList.add('vg-tooltip');

      const tooltipContainer = document.fullscreenElement ?? document.body;
      tooltipContainer.appendChild(this.el);
    }

    // hide tooltip for null, undefined, or empty string values
    if (value == null || value === '') {
      this.el.classList.remove('visible', `${this.options.theme}-theme`);
      return;
    }

    // set the tooltip content
    this.el.innerHTML = this.options.formatTooltip(
      value,
      this.options.sanitize,
      this.options.maxDepth,
      this.options.baseURL,
    );

    // make the tooltip visible
    this.el.classList.add('visible', `${this.options.theme}-theme`);

    const {x, y} = calculatePosition(
      event,
      this.el.getBoundingClientRect(),
      this.options.offsetX,
      this.options.offsetY,
    );

    this.el.style.top = `${y}px`;
    this.el.style.left = `${x}px`;
  }
}

'use strict';

/* global requestIdleCallback, requestAnimationFrame, cancelIdleCallback, cancelAnimationFrame */

/**
 * Utility module to work with EcmaScript's event loop.
 *
 * @module eventloop
 */

/**
 * @type {Array<function>}
 */
let queue = [];

const _runQueue = () => {
  for (let i = 0; i < queue.length; i++) {
    queue[i]();
  }
  queue = [];
};

/**
 * @param {function():void} f
 */
const enqueue = f => {
  queue.push(f);
  if (queue.length === 1) {
    setTimeout(_runQueue, 0);
  }
};

/**
 * @typedef {Object} TimeoutObject
 * @property {function} TimeoutObject.destroy
 */

/**
 * @param {function(number):void} clearFunction
 */
const createTimeoutClass = clearFunction => class TT {
  /**
   * @param {number} timeoutId
   */
  constructor (timeoutId) {
    this._ = timeoutId;
  }

  destroy () {
    clearFunction(this._);
  }
};

const Timeout = createTimeoutClass(clearTimeout);

/**
 * @param {number} timeout
 * @param {function} callback
 * @return {TimeoutObject}
 */
const timeout = (timeout, callback) => new Timeout(setTimeout(callback, timeout));

const Interval = createTimeoutClass(clearInterval);

/**
 * @param {number} timeout
 * @param {function} callback
 * @return {TimeoutObject}
 */
const interval = (timeout, callback) => new Interval(setInterval(callback, timeout));

/* c8 ignore next */
const Animation = createTimeoutClass(arg => typeof requestAnimationFrame !== 'undefined' && cancelAnimationFrame(arg));

/**
 * @param {function(number):void} cb
 * @return {TimeoutObject}
 */
/* c8 ignore next */
const animationFrame = cb => typeof requestAnimationFrame === 'undefined' ? timeout(0, cb) : new Animation(requestAnimationFrame(cb));

/* c8 ignore next */
// @ts-ignore
const Idle = createTimeoutClass(arg => typeof cancelIdleCallback !== 'undefined' && cancelIdleCallback(arg));

/**
 * Note: this is experimental and is probably only useful in browsers.
 *
 * @param {function} cb
 * @return {TimeoutObject}
 */
/* c8 ignore next 2 */
// @ts-ignore
const idleCallback = cb => typeof requestIdleCallback !== 'undefined' ? new Idle(requestIdleCallback(cb)) : timeout(1000, cb);

/**
 * @param {number} timeout Timeout of the debounce action
 * @return {function(function():void):void}
 */
const createDebouncer = timeout => {
  let timer = -1;
  return f => {
    clearTimeout(timer);
    if (f) {
      timer = /** @type {any} */ (setTimeout(f, timeout));
    }
  }
};

var eventloop = /*#__PURE__*/Object.freeze({
  __proto__: null,
  enqueue: enqueue,
  timeout: timeout,
  interval: interval,
  Animation: Animation,
  animationFrame: animationFrame,
  idleCallback: idleCallback,
  createDebouncer: createDebouncer
});

exports.Animation = Animation;
exports.animationFrame = animationFrame;
exports.createDebouncer = createDebouncer;
exports.enqueue = enqueue;
exports.eventloop = eventloop;
exports.idleCallback = idleCallback;
exports.interval = interval;
exports.timeout = timeout;
//# sourceMappingURL=eventloop-d0571621.cjs.map

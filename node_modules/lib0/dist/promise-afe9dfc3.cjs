'use strict';

var time = require('./time-bc2081b9.cjs');

/**
 * Utility helpers to work with promises.
 *
 * @module promise
 */

/**
 * @template T
 * @callback PromiseResolve
 * @param {T|PromiseLike<T>} [result]
 */

/**
 * @template T
 * @param {function(PromiseResolve<T>,function(Error):void):any} f
 * @return {Promise<T>}
 */
const create = f => /** @type {Promise<T>} */ (new Promise(f));

/**
 * @param {function(function():void,function(Error):void):void} f
 * @return {Promise<void>}
 */
const createEmpty = f => new Promise(f);

/**
 * `Promise.all` wait for all promises in the array to resolve and return the result
 * @template {unknown[] | []} PS
 *
 * @param {PS} ps
 * @return {Promise<{ -readonly [P in keyof PS]: Awaited<PS[P]> }>}
 */
const all = Promise.all.bind(Promise);

/**
 * @param {Error} [reason]
 * @return {Promise<never>}
 */
const reject = reason => Promise.reject(reason);

/**
 * @template T
 * @param {T|void} res
 * @return {Promise<T|void>}
 */
const resolve = res => Promise.resolve(res);

/**
 * @template T
 * @param {T} res
 * @return {Promise<T>}
 */
const resolveWith = res => Promise.resolve(res);

/**
 * @todo Next version, reorder parameters: check, [timeout, [intervalResolution]]
 *
 * @param {number} timeout
 * @param {function():boolean} check
 * @param {number} [intervalResolution]
 * @return {Promise<void>}
 */
const until = (timeout, check, intervalResolution = 10) => create((resolve, reject) => {
  const startTime = time.getUnixTime();
  const hasTimeout = timeout > 0;
  const untilInterval = () => {
    if (check()) {
      clearInterval(intervalHandle);
      resolve();
    } else if (hasTimeout) {
      /* c8 ignore else */
      if (time.getUnixTime() - startTime > timeout) {
        clearInterval(intervalHandle);
        reject(new Error('Timeout'));
      }
    }
  };
  const intervalHandle = setInterval(untilInterval, intervalResolution);
});

/**
 * @param {number} timeout
 * @return {Promise<undefined>}
 */
const wait = timeout => create((resolve, reject) => setTimeout(resolve, timeout));

/**
 * Checks if an object is a promise using ducktyping.
 *
 * Promises are often polyfilled, so it makes sense to add some additional guarantees if the user of this
 * library has some insane environment where global Promise objects are overwritten.
 *
 * @param {any} p
 * @return {boolean}
 */
const isPromise = p => p instanceof Promise || (p && p.then && p.catch && p.finally);

var promise = /*#__PURE__*/Object.freeze({
  __proto__: null,
  create: create,
  createEmpty: createEmpty,
  all: all,
  reject: reject,
  resolve: resolve,
  resolveWith: resolveWith,
  until: until,
  wait: wait,
  isPromise: isPromise
});

exports.all = all;
exports.create = create;
exports.createEmpty = createEmpty;
exports.isPromise = isPromise;
exports.promise = promise;
exports.reject = reject;
exports.resolve = resolve;
exports.resolveWith = resolveWith;
exports.until = until;
exports.wait = wait;
//# sourceMappingURL=promise-afe9dfc3.cjs.map

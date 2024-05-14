'use strict';

/**
 * Utility functions for working with EcmaScript objects.
 *
 * @module object
 */

/**
 * @return {Object<string,any>} obj
 */
const create = () => Object.create(null);

/**
 * Object.assign
 */
const assign = Object.assign;

/**
 * @param {Object<string,any>} obj
 */
const keys = Object.keys;

/**
 * @template V
 * @param {{[k:string]:V}} obj
 * @param {function(V,string):any} f
 */
const forEach = (obj, f) => {
  for (const key in obj) {
    f(obj[key], key);
  }
};

/**
 * @todo implement mapToArray & map
 *
 * @template R
 * @param {Object<string,any>} obj
 * @param {function(any,string):R} f
 * @return {Array<R>}
 */
const map = (obj, f) => {
  const results = [];
  for (const key in obj) {
    results.push(f(obj[key], key));
  }
  return results
};

/**
 * @param {Object<string,any>} obj
 * @return {number}
 */
const length = obj => keys(obj).length;

/**
 * @param {Object<string,any>} obj
 * @param {function(any,string):boolean} f
 * @return {boolean}
 */
const some = (obj, f) => {
  for (const key in obj) {
    if (f(obj[key], key)) {
      return true
    }
  }
  return false
};

/**
 * @param {Object|undefined} obj
 */
const isEmpty = obj => {
  // eslint-disable-next-line
  for (const _k in obj) {
    return false
  }
  return true
};

/**
 * @param {Object<string,any>} obj
 * @param {function(any,string):boolean} f
 * @return {boolean}
 */
const every = (obj, f) => {
  for (const key in obj) {
    if (!f(obj[key], key)) {
      return false
    }
  }
  return true
};

/**
 * Calls `Object.prototype.hasOwnProperty`.
 *
 * @param {any} obj
 * @param {string|symbol} key
 * @return {boolean}
 */
const hasProperty = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

/**
 * @param {Object<string,any>} a
 * @param {Object<string,any>} b
 * @return {boolean}
 */
const equalFlat = (a, b) => a === b || (length(a) === length(b) && every(a, (val, key) => (val !== undefined || hasProperty(b, key)) && b[key] === val));

var object = /*#__PURE__*/Object.freeze({
  __proto__: null,
  create: create,
  assign: assign,
  keys: keys,
  forEach: forEach,
  map: map,
  length: length,
  some: some,
  isEmpty: isEmpty,
  every: every,
  hasProperty: hasProperty,
  equalFlat: equalFlat
});

exports.assign = assign;
exports.create = create;
exports.equalFlat = equalFlat;
exports.every = every;
exports.forEach = forEach;
exports.hasProperty = hasProperty;
exports.isEmpty = isEmpty;
exports.keys = keys;
exports.length = length;
exports.map = map;
exports.object = object;
exports.some = some;
//# sourceMappingURL=object-fecf6a7b.cjs.map

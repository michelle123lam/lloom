'use strict';

var set = require('./set-0f209abb.cjs');

/**
 * Utility module to work with Arrays.
 *
 * @module array
 */

/**
 * Return the last element of an array. The element must exist
 *
 * @template L
 * @param {ArrayLike<L>} arr
 * @return {L}
 */
const last = arr => arr[arr.length - 1];

/**
 * @template C
 * @return {Array<C>}
 */
const create = () => /** @type {Array<C>} */ ([]);

/**
 * @template D
 * @param {Array<D>} a
 * @return {Array<D>}
 */
const copy = a => /** @type {Array<D>} */ (a.slice());

/**
 * Append elements from src to dest
 *
 * @template M
 * @param {Array<M>} dest
 * @param {Array<M>} src
 */
const appendTo = (dest, src) => {
  for (let i = 0; i < src.length; i++) {
    dest.push(src[i]);
  }
};

/**
 * Transforms something array-like to an actual Array.
 *
 * @function
 * @template T
 * @param {ArrayLike<T>|Iterable<T>} arraylike
 * @return {T}
 */
const from = Array.from;

/**
 * True iff condition holds on every element in the Array.
 *
 * @function
 * @template ITEM
 * @template {ArrayLike<ITEM>} ARR
 *
 * @param {ARR} arr
 * @param {function(ITEM, number, ARR):boolean} f
 * @return {boolean}
 */
const every = (arr, f) => {
  for (let i = 0; i < arr.length; i++) {
    if (!f(arr[i], i, arr)) {
      return false
    }
  }
  return true
};

/**
 * True iff condition holds on some element in the Array.
 *
 * @function
 * @template S
 * @template {ArrayLike<S>} ARR
 * @param {ARR} arr
 * @param {function(S, number, ARR):boolean} f
 * @return {boolean}
 */
const some = (arr, f) => {
  for (let i = 0; i < arr.length; i++) {
    if (f(arr[i], i, arr)) {
      return true
    }
  }
  return false
};

/**
 * @template ELEM
 *
 * @param {ArrayLike<ELEM>} a
 * @param {ArrayLike<ELEM>} b
 * @return {boolean}
 */
const equalFlat = (a, b) => a.length === b.length && every(a, (item, index) => item === b[index]);

/**
 * @template ELEM
 * @param {Array<Array<ELEM>>} arr
 * @return {Array<ELEM>}
 */
const flatten = arr => fold(arr, /** @type {Array<ELEM>} */ ([]), (acc, val) => acc.concat(val));

/**
 * @template T
 * @param {number} len
 * @param {function(number, Array<T>):T} f
 * @return {Array<T>}
 */
const unfold = (len, f) => {
  const array = new Array(len);
  for (let i = 0; i < len; i++) {
    array[i] = f(i, array);
  }
  return array
};

/**
 * @template T
 * @template RESULT
 * @param {Array<T>} arr
 * @param {RESULT} seed
 * @param {function(RESULT, T, number):RESULT} folder
 */
const fold = (arr, seed, folder) => arr.reduce(folder, seed);

const isArray = Array.isArray;

/**
 * @template T
 * @param {Array<T>} arr
 * @return {Array<T>}
 */
const unique = arr => from(set.from(arr));

/**
 * @template T
 * @template M
 * @param {ArrayLike<T>} arr
 * @param {function(T):M} mapper
 * @return {Array<T>}
 */
const uniqueBy = (arr, mapper) => {
  /**
   * @type {Set<M>}
   */
  const happened = set.create();
  /**
   * @type {Array<T>}
   */
  const result = [];
  for (let i = 0; i < arr.length; i++) {
    const el = arr[i];
    const mapped = mapper(el);
    if (!happened.has(mapped)) {
      happened.add(mapped);
      result.push(el);
    }
  }
  return result
};

/**
 * @template {ArrayLike<any>} ARR
 * @template {function(ARR extends ArrayLike<infer T> ? T : never, number, ARR):any} MAPPER
 * @param {ARR} arr
 * @param {MAPPER} mapper
 * @return {Array<MAPPER extends function(...any): infer M ? M : never>}
 */
const map = (arr, mapper) => {
  /**
   * @type {Array<any>}
   */
  const res = Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    res[i] = mapper(/** @type {any} */ (arr[i]), i, /** @type {any} */ (arr));
  }
  return /** @type {any} */ (res)
};

var array = /*#__PURE__*/Object.freeze({
  __proto__: null,
  last: last,
  create: create,
  copy: copy,
  appendTo: appendTo,
  from: from,
  every: every,
  some: some,
  equalFlat: equalFlat,
  flatten: flatten,
  unfold: unfold,
  fold: fold,
  isArray: isArray,
  unique: unique,
  uniqueBy: uniqueBy,
  map: map
});

exports.appendTo = appendTo;
exports.array = array;
exports.copy = copy;
exports.create = create;
exports.equalFlat = equalFlat;
exports.every = every;
exports.flatten = flatten;
exports.fold = fold;
exports.from = from;
exports.isArray = isArray;
exports.last = last;
exports.map = map;
exports.some = some;
exports.unfold = unfold;
exports.unique = unique;
exports.uniqueBy = uniqueBy;
//# sourceMappingURL=array-704ca50e.cjs.map

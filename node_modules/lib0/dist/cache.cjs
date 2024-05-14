'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var list = require('./list.cjs');
var map = require('./map-0dabcc55.cjs');
var time = require('./time-bc2081b9.cjs');
require('./function-314fdc56.cjs');
require('./array-704ca50e.cjs');
require('./set-0f209abb.cjs');
require('./object-fecf6a7b.cjs');
require('./error-8582d695.cjs');
require('./metric.cjs');
require('./math-08e068f9.cjs');

/* eslint-env browser */

/**
 * @template K, V
 *
 * @implements {list.ListNode}
 */
class Entry {
  /**
   * @param {K} key
   * @param {V | Promise<V>} val
   */
  constructor (key, val) {
    /**
     * @type {this | null}
     */
    this.prev = null;
    /**
     * @type {this | null}
     */
    this.next = null;
    this.created = time.getUnixTime();
    this.val = val;
    this.key = key;
  }
}

/**
 * @template K, V
 */
class Cache {
  /**
   * @param {number} timeout
   */
  constructor (timeout) {
    this.timeout = timeout;
    /**
     * @type list.List<Entry<K, V>>
     */
    this._q = list.create();
    /**
     * @type {Map<K, Entry<K, V>>}
     */
    this._map = map.create();
  }
}

/**
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @return {number} Returns the current timestamp
 */
const removeStale = cache => {
  const now = time.getUnixTime();
  const q = cache._q;
  while (q.start && now - q.start.created > cache.timeout) {
    cache._map.delete(q.start.key);
    list.popFront(q);
  }
  return now
};

/**
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @param {K} key
 * @param {V} value
 */
const set = (cache, key, value) => {
  const now = removeStale(cache);
  const q = cache._q;
  const n = cache._map.get(key);
  if (n) {
    list.removeNode(q, n);
    list.pushEnd(q, n);
    n.created = now;
    n.val = value;
  } else {
    const node = new Entry(key, value);
    list.pushEnd(q, node);
    cache._map.set(key, node);
  }
};

/**
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @param {K} key
 * @return {Entry<K, V> | undefined}
 */
const getNode = (cache, key) => {
  removeStale(cache);
  const n = cache._map.get(key);
  if (n) {
    return n
  }
};

/**
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @param {K} key
 * @return {V | undefined}
 */
const get = (cache, key) => {
  const n = getNode(cache, key);
  return n && !(n.val instanceof Promise) ? n.val : undefined
};

/**
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @param {K} key
 */
const refreshTimeout = (cache, key) => {
  const now = time.getUnixTime();
  const q = cache._q;
  const n = cache._map.get(key);
  if (n) {
    list.removeNode(q, n);
    list.pushEnd(q, n);
    n.created = now;
  }
};

/**
 * Works well in conjunktion with setIfUndefined which has an async init function.
 * Using getAsync & setIfUndefined ensures that the init function is only called once.
 *
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @param {K} key
 * @return {V | Promise<V> | undefined}
 */
const getAsync = (cache, key) => {
  const n = getNode(cache, key);
  return n ? n.val : undefined
};

/**
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @param {K} key
 */
const remove = (cache, key) => {
  const n = cache._map.get(key);
  if (n) {
    list.removeNode(cache._q, n);
    cache._map.delete(key);
    return n.val && !(n.val instanceof Promise) ? n.val : undefined
  }
};

/**
 * @template K, V
 *
 * @param {Cache<K, V>} cache
 * @param {K} key
 * @param {function():Promise<V>} init
 * @param {boolean} removeNull Optional argument that automatically removes values that resolve to null/undefined from the cache.
 * @return {Promise<V> | V}
 */
const setIfUndefined = (cache, key, init, removeNull = false) => {
  removeStale(cache);
  const q = cache._q;
  const n = cache._map.get(key);
  if (n) {
    return n.val
  } else {
    const p = init();
    const node = new Entry(key, p);
    list.pushEnd(q, node);
    cache._map.set(key, node);
    p.then(v => {
      if (p === node.val) {
        node.val = v;
      }
      if (removeNull && v == null) {
        remove(cache, key);
      }
    });
    return p
  }
};

/**
 * @param {number} timeout
 */
const create = timeout => new Cache(timeout);

exports.Cache = Cache;
exports.create = create;
exports.get = get;
exports.getAsync = getAsync;
exports.refreshTimeout = refreshTimeout;
exports.remove = remove;
exports.removeStale = removeStale;
exports.set = set;
exports.setIfUndefined = setIfUndefined;
//# sourceMappingURL=cache.cjs.map

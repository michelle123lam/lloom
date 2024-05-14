'use strict';

/**
 * Utility module to work with EcmaScript Symbols.
 *
 * @module symbol
 */

/**
 * Return fresh symbol.
 *
 * @return {Symbol}
 */
const create = Symbol;

/**
 * @param {any} s
 * @return {boolean}
 */
const isSymbol = s => typeof s === 'symbol';

var symbol = /*#__PURE__*/Object.freeze({
	__proto__: null,
	create: create,
	isSymbol: isSymbol
});

exports.create = create;
exports.isSymbol = isSymbol;
exports.symbol = symbol;
//# sourceMappingURL=symbol-c5caa724.cjs.map

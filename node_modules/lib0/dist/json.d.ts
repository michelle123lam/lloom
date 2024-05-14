/**
 * JSON utility functions.
 *
 * @module json
 */
/**
 * Transform JavaScript object to JSON.
 *
 * @param {any} object
 * @return {string}
 */
export const stringify: {
    (value: any, replacer?: ((this: any, key: string, value: any) => any) | undefined, space?: string | number | undefined): string;
    (value: any, replacer?: (string | number)[] | null | undefined, space?: string | number | undefined): string;
};
/**
 * Parse JSON object.
 *
 * @param {string} json
 * @return {any}
 */
export const parse: (text: string, reviver?: ((this: any, key: string, value: any) => any) | undefined) => any;
//# sourceMappingURL=json.d.ts.map
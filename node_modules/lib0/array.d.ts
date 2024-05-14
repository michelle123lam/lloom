export function last<L>(arr: ArrayLike<L>): L;
export function create<C>(): C[];
export function copy<D>(a: D[]): D[];
export function appendTo<M>(dest: M[], src: M[]): void;
/**
 * Transforms something array-like to an actual Array.
 *
 * @function
 * @template T
 * @param {ArrayLike<T>|Iterable<T>} arraylike
 * @return {T}
 */
export const from: {
    <T_1>(arrayLike: ArrayLike<T_1>): T_1[];
    <T_2, U>(arrayLike: ArrayLike<T_2>, mapfn: (v: T_2, k: number) => U, thisArg?: any): U[];
    <T_3>(iterable: Iterable<T_3> | ArrayLike<T_3>): T_3[];
    <T_4, U_1>(iterable: Iterable<T_4> | ArrayLike<T_4>, mapfn: (v: T_4, k: number) => U_1, thisArg?: any): U_1[];
};
export function every<ITEM, ARR extends ArrayLike<ITEM>>(arr: ARR, f: (arg0: ITEM, arg1: number, arg2: ARR) => boolean): boolean;
export function some<S, ARR extends ArrayLike<S>>(arr: ARR, f: (arg0: S, arg1: number, arg2: ARR) => boolean): boolean;
export function equalFlat<ELEM>(a: ArrayLike<ELEM>, b: ArrayLike<ELEM>): boolean;
export function flatten<ELEM>(arr: ELEM[][]): ELEM[];
export function unfold<T_1>(len: number, f: (arg0: number, arg1: T_1[]) => T_1): T_1[];
export function fold<T_1, RESULT>(arr: T_1[], seed: RESULT, folder: (arg0: RESULT, arg1: T_1, arg2: number) => RESULT): RESULT;
export const isArray: (arg: any) => arg is any[];
export function unique<T_1>(arr: T_1[]): T_1[];
export function uniqueBy<T_1, M>(arr: ArrayLike<T_1>, mapper: (arg0: T_1) => M): T_1[];
export function map<ARR extends ArrayLike<any>, MAPPER extends (arg0: ARR extends ArrayLike<infer T_1> ? T_1 : never, arg1: number, arg2: ARR) => any>(arr: ARR, mapper: MAPPER): (MAPPER extends (...arg0: any[]) => infer M ? M : never)[];
//# sourceMappingURL=array.d.ts.map
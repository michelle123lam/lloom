export function create(): {
    [x: string]: any;
};
/**
 * Object.assign
 */
export const assign: {
    <T extends {}, U>(target: T, source: U): T & U;
    <T_1 extends {}, U_1, V>(target: T_1, source1: U_1, source2: V): T_1 & U_1 & V;
    <T_2 extends {}, U_2, V_1, W>(target: T_2, source1: U_2, source2: V_1, source3: W): T_2 & U_2 & V_1 & W;
    (target: object, ...sources: any[]): any;
};
/**
 * @param {Object<string,any>} obj
 */
export const keys: {
    (o: object): string[];
    (o: {}): string[];
};
export function forEach<V>(obj: {
    [k: string]: V;
}, f: (arg0: V, arg1: string) => any): void;
export function map<R>(obj: {
    [x: string]: any;
}, f: (arg0: any, arg1: string) => R): R[];
export function length(obj: {
    [x: string]: any;
}): number;
export function some(obj: {
    [x: string]: any;
}, f: (arg0: any, arg1: string) => boolean): boolean;
export function isEmpty(obj: Object | undefined): boolean;
export function every(obj: {
    [x: string]: any;
}, f: (arg0: any, arg1: string) => boolean): boolean;
export function hasProperty(obj: any, key: string | symbol): boolean;
export function equalFlat(a: {
    [x: string]: any;
}, b: {
    [x: string]: any;
}): boolean;
//# sourceMappingURL=object.d.ts.map
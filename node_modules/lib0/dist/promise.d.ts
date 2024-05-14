export function create<T>(f: (arg0: PromiseResolve<T>, arg1: (arg0: Error) => void) => any): Promise<T>;
export function createEmpty(f: (arg0: () => void, arg1: (arg0: Error) => void) => void): Promise<void>;
/**
 * `Promise.all` wait for all promises in the array to resolve and return the result
 * @template {unknown[] | []} PS
 *
 * @param {PS} ps
 * @return {Promise<{ -readonly [P in keyof PS]: Awaited<PS[P]> }>}
 */
export const all: {
    <T>(values: Iterable<T | PromiseLike<T>>): Promise<Awaited<T>[]>;
    <T_1 extends [] | readonly unknown[]>(values: T_1): Promise<{ -readonly [P in keyof T_1]: Awaited<T_1[P]>; }>;
};
export function reject(reason?: Error | undefined): Promise<never>;
export function resolve<T>(res: void | T): Promise<void | T>;
export function resolveWith<T>(res: T): Promise<T>;
export function until(timeout: number, check: () => boolean, intervalResolution?: number | undefined): Promise<void>;
export function wait(timeout: number): Promise<undefined>;
export function isPromise(p: any): boolean;
export type PromiseResolve<T> = (result?: T | PromiseLike<T> | undefined) => any;
//# sourceMappingURL=promise.d.ts.map
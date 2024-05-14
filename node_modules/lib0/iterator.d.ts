export function mapIterator<T, R>(iterator: Iterator<T, any, undefined>, f: (arg0: T) => R): IterableIterator<R>;
export function createIterator<T>(next: () => IteratorResult<T, any>): IterableIterator<T>;
export function iteratorFilter<T>(iterator: Iterator<T, any, undefined>, filter: (arg0: T) => boolean): IterableIterator<T>;
export function iteratorMap<T, M>(iterator: Iterator<T, any, undefined>, fmap: (arg0: T) => M): IterableIterator<M | undefined>;
//# sourceMappingURL=iterator.d.ts.map
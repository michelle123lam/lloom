export class ListNode {
    /**
     * @type {this|null}
     */
    next: ListNode | null;
    /**
     * @type {this|null}
     */
    prev: ListNode | null;
}
/**
 * @template {ListNode} N
 */
export class List<N extends ListNode> {
    /**
     * @type {N | null}
     */
    start: N | null;
    /**
     * @type {N | null}
     */
    end: N | null;
    len: number;
}
export function create<N extends ListNode>(): List<N>;
export function isEmpty<N extends ListNode>(queue: List<N>): boolean;
export function remove<N extends ListNode>(queue: List<N>, node: N): N;
export function removeNode<N extends ListNode>(queue: List<N>, node: N): N;
export function insertBetween<N extends ListNode>(queue: List<N>, left: N | null, right: N | null, node: N): void;
export function replace<N extends ListNode>(queue: List<N>, node: N, newNode: N): void;
export function pushEnd<N extends ListNode>(queue: List<N>, n: N): void;
export function pushFront<N extends ListNode>(queue: List<N>, n: N): void;
export function popFront<N extends ListNode>(list: List<N>): N | null;
export function popEnd<N extends ListNode>(list: List<N>): N | null;
export function map<N extends ListNode, M>(list: List<N>, f: (arg0: N) => M): M[];
export function toArray<N extends ListNode>(list: List<N>): N[];
export function forEach<N extends ListNode, M>(list: List<N>, f: (arg0: N) => M): void;
//# sourceMappingURL=list.d.ts.map
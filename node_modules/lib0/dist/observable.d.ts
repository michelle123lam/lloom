/**
 * Handles named events.
 * @experimental
 *
 * This is basically a (better typed) duplicate of Observable, which will replace Observable in the
 * next release.
 *
 * @template {{[key in keyof EVENTS]: function(...any):void}} EVENTS
 */
export class ObservableV2<EVENTS extends { [key in keyof EVENTS]: (...arg0: any[]) => void; }> {
    /**
     * Some desc.
     * @type {Map<string, Set<any>>}
     */
    _observers: Map<string, Set<any>>;
    /**
     * @template {keyof EVENTS & string} NAME
     * @param {NAME} name
     * @param {EVENTS[NAME]} f
     */
    on<NAME extends keyof EVENTS & string>(name: NAME, f: EVENTS[NAME]): EVENTS[NAME];
    /**
     * @template {keyof EVENTS & string} NAME
     * @param {NAME} name
     * @param {EVENTS[NAME]} f
     */
    once<NAME_1 extends keyof EVENTS & string>(name: NAME_1, f: EVENTS[NAME_1]): void;
    /**
     * @template {keyof EVENTS & string} NAME
     * @param {NAME} name
     * @param {EVENTS[NAME]} f
     */
    off<NAME_2 extends keyof EVENTS & string>(name: NAME_2, f: EVENTS[NAME_2]): void;
    /**
     * Emit a named event. All registered event listeners that listen to the
     * specified name will receive the event.
     *
     * @todo This should catch exceptions
     *
     * @template {keyof EVENTS & string} NAME
     * @param {NAME} name The event name.
     * @param {Parameters<EVENTS[NAME]>} args The arguments that are applied to the event listener.
     */
    emit<NAME_3 extends keyof EVENTS & string>(name: NAME_3, args: Parameters<EVENTS[NAME_3]>): void;
    destroy(): void;
}
/**
 * Handles named events.
 *
 * @deprecated
 * @template N
 */
export class Observable<N> {
    /**
     * Some desc.
     * @type {Map<N, any>}
     */
    _observers: Map<N, any>;
    /**
     * @param {N} name
     * @param {function} f
     */
    on(name: N, f: Function): void;
    /**
     * @param {N} name
     * @param {function} f
     */
    once(name: N, f: Function): void;
    /**
     * @param {N} name
     * @param {function} f
     */
    off(name: N, f: Function): void;
    /**
     * Emit a named event. All registered event listeners that listen to the
     * specified name will receive the event.
     *
     * @todo This should catch exceptions
     *
     * @param {N} name The event name.
     * @param {Array<any>} args The arguments that are applied to the event listener.
     */
    emit(name: N, args: Array<any>): void;
    destroy(): void;
}
//# sourceMappingURL=observable.d.ts.map
/* -----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
export function convertYMapEventToMapChange(event) {
    let changes = new Map();
    event.changes.keys.forEach((event, key) => {
        changes.set(key, {
            action: event.action,
            oldValue: event.oldValue,
            newValue: this.ymeta.get(key)
        });
    });
    return changes;
}
/**
 * Creates a mutual exclude function with the following property:
 *
 * ```js
 * const mutex = createMutex()
 * mutex(() => {
 *   // This function is immediately executed
 *   mutex(() => {
 *     // This function is not executed, as the mutex is already active.
 *   })
 * })
 * ```
 */
export const createMutex = () => {
    let token = true;
    return (f) => {
        if (token) {
            token = false;
            try {
                f();
            }
            finally {
                token = true;
            }
        }
    };
};
//# sourceMappingURL=utils.js.map
/**
 * The namespace for `PageConfig` functions.
 */
export declare namespace PageConfig {
    /**
     * Get global configuration data for the Jupyter application.
     *
     * @param name - The name of the configuration option.
     *
     * @returns The config value or an empty string if not found.
     *
     * #### Notes
     * All values are treated as strings.
     * For browser based applications, it is assumed that the page HTML
     * includes a script tag with the id `jupyter-config-data` containing the
     * configuration as valid JSON.  In order to support the classic Notebook,
     * we fall back on checking for `body` data of the given `name`.
     *
     * For node applications, it is assumed that the process was launched
     * with a `--jupyter-config-data` option pointing to a JSON settings
     * file.
     */
    function getOption(name: string): string;
    /**
     * Set global configuration data for the Jupyter application.
     *
     * @param name - The name of the configuration option.
     * @param value - The value to set the option to.
     *
     * @returns The last config value or an empty string if it doesn't exist.
     */
    function setOption(name: string, value: string): string;
    /**
     * Get the base url for a Jupyter application, or the base url of the page.
     */
    function getBaseUrl(): string;
    /**
     * Get the tree url for a JupyterLab application.
     */
    function getTreeUrl(): string;
    /**
     * Get the base url for sharing links (usually baseUrl)
     */
    function getShareUrl(): string;
    /**
     * Get the tree url for shareable links.
     * Usually the same as treeUrl,
     * but overrideable e.g. when sharing with JupyterHub.
     */
    function getTreeShareUrl(): string;
    /**
     * Create a new URL given an optional mode and tree path.
     *
     * This is used to create URLS when the mode or tree path change as the user
     * changes mode or the current document in the main area. If fields in
     * options are omitted, the value in PageConfig will be used.
     *
     * @param options - IGetUrlOptions for the new path.
     */
    function getUrl(options: IGetUrlOptions): string;
    const defaultWorkspace: string;
    /**
     * Options for getUrl
     */
    interface IGetUrlOptions {
        /**
         * The optional mode as a string 'single-document' or 'multiple-document'. If
         * the mode argument is missing, it will be provided from the PageConfig.
         */
        mode?: string;
        /**
         * The optional workspace as a string. If this argument is missing, the value will
         * be pulled from PageConfig. To use the default workspace (no /workspaces/<name>
         * URL segment will be included) pass the string PageConfig.defaultWorkspace.
         */
        workspace?: string;
        /**
         * Whether the url is meant to be shared or not; default false.
         */
        toShare?: boolean;
        /**
         * The optional tree path as as string. If treePath is not provided it will be
         * provided from the PageConfig. If an empty string, the resulting path will not
         * contain a tree portion.
         */
        treePath?: string;
    }
    /**
     * Get the base websocket url for a Jupyter application, or an empty string.
     */
    function getWsUrl(baseUrl?: string): string;
    /**
     * Returns the URL converting this notebook to a certain
     * format with nbconvert.
     */
    function getNBConvertURL({ path, format, download }: {
        path: string;
        format: string;
        download: boolean;
    }): string;
    /**
     * Get the authorization token for a Jupyter application.
     */
    function getToken(): string;
    /**
     * Get the Notebook version info [major, minor, patch].
     */
    function getNotebookVersion(): [number, number, number];
    /**
     * The namespace for page config `Extension` functions.
     */
    namespace Extension {
        /**
         * The collection of deferred extensions in page config.
         */
        const deferred: string[];
        /**
         * The collection of disabled extensions in page config.
         */
        const disabled: string[];
        /**
         * Returns whether a plugin is deferred.
         *
         * @param id - The plugin ID.
         */
        function isDeferred(id: string): boolean;
        /**
         * Returns whether a plugin is disabled.
         *
         * @param id - The plugin ID.
         */
        function isDisabled(id: string): boolean;
    }
}

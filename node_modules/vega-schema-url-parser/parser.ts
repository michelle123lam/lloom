/**
 * Parse a vega schema url into library and version.
 */
export default function(url: string) {
    const regex = /schema\/([\w-]+)\/([\w\.\-]+)\.json$/g;
    const [library, version] = regex.exec(url)!.slice(1, 3);
    return {library: library as 'vega' | 'vega-lite', version};
}

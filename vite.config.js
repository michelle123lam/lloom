import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
	plugins: [
		svelte({ hot: false }),
		anywidget(),
	],
	server: {
		hmr: {
			overlay: false,
		},
	},
	build: {
        outDir: "text_lloom/src/text_lloom/static",
        emptyOutDir: false,
        lib: {
            entry: {
                index: "src/index.js",
            },
            formats: ["es"],
        },
        rollupOptions: {
            output: {
                assetFileNames: "index.[ext]",
            }
        }
    },
});

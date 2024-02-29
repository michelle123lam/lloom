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
        outDir: "lloom_ai/src/lloom_ai/static",
        lib: {
            entry: {
                index_select: "src/index_select.js",
            },
            formats: ["es"],
        },
        rollupOptions: {
            output: {
                assetFileNames: "index_select.[ext]",
            }
        }
    },
});

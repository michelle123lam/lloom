/*
  @license
	Rollup.js v4.14.0
	Wed, 03 Apr 2024 05:22:15 GMT - commit 5abe71bd5bae3423b4e2ee80207c871efde20253

	https://github.com/rollup/rollup

	Released under the MIT License.
*/
export { version as VERSION, defineConfig, rollup, watch } from './shared/node-entry.js';
import './shared/parseAst.js';
import '../native.js';
import 'node:path';
import 'path';
import 'node:process';
import 'node:perf_hooks';
import 'node:fs/promises';
import 'tty';

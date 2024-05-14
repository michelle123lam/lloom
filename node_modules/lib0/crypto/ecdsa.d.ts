export function sign(key: CryptoKey, data: Uint8Array): PromiseLike<Uint8Array>;
export function verify(key: CryptoKey, signature: Uint8Array, data: Uint8Array): PromiseLike<boolean>;
export function generateKeyPair({ extractable, usages }?: {
    extractable?: boolean | undefined;
    usages?: Usages | undefined;
}): Promise<CryptoKeyPair>;
export function importKeyJwk(jwk: any, { extractable, usages }?: {
    extractable?: boolean | undefined;
    usages?: Usages | undefined;
}): Promise<CryptoKey>;
export function importKeyRaw(raw: any, { extractable, usages }?: {
    extractable?: boolean | undefined;
    usages?: Usages | undefined;
}): Promise<CryptoKey>;
export type Usages = Array<'sign' | 'verify'>;
export { exportKeyJwk, exportKeyRaw } from "./common.js";
//# sourceMappingURL=ecdsa.d.ts.map
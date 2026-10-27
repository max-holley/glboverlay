import type { ModelSettings } from './settings';

export interface StoredFile {
  name: string;
  blob: Blob;
}

export interface ModelMeta {
  id: string;
  name: string;
  /** file name of the primary .glb/.gltf/.vrm */
  main: string;
  size: number;
  date: number;
  thumbnail?: Blob;
  settings: ModelSettings;
}

const DB_NAME = 'glboverlay';
const META = 'models';
const FILES = 'files';

export class StorageFullError extends Error {}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore(META, { keyPath: 'id' });
      db.createObjectStore(FILES);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB unavailable'));
  });
}

function wrap<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function done(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error);
    tx.onerror = () => reject(tx.error);
  });
}

function isQuota(e: unknown): boolean {
  return e instanceof DOMException && (e.name === 'QuotaExceededError' || e.code === 22);
}

export class ModelStore {
  private db: Promise<IDBDatabase> = open();

  async list(): Promise<ModelMeta[]> {
    const db = await this.db;
    const all = await wrap(db.transaction(META).objectStore(META).getAll() as IDBRequest<ModelMeta[]>);
    return all.sort((a, b) => b.date - a.date);
  }

  async get(id: string): Promise<ModelMeta | undefined> {
    const db = await this.db;
    return wrap(db.transaction(META).objectStore(META).get(id) as IDBRequest<ModelMeta | undefined>);
  }

  async getFiles(id: string): Promise<StoredFile[]> {
    const db = await this.db;
    const files = await wrap(db.transaction(FILES).objectStore(FILES).get(id) as IDBRequest<StoredFile[] | undefined>);
    if (!files) throw new Error('Stored model data is missing.');
    return files;
  }

  async add(meta: ModelMeta, files: StoredFile[]): Promise<void> {
    const db = await this.db;
    try {
      const tx = db.transaction([META, FILES], 'readwrite');
      tx.objectStore(FILES).put(files, meta.id);
      tx.objectStore(META).put(meta);
      await done(tx);
    } catch (e) {
      if (isQuota(e)) throw new StorageFullError('Browser storage is full.');
      throw e;
    }
  }

  async updateMeta(meta: ModelMeta): Promise<void> {
    const db = await this.db;
    try {
      const tx = db.transaction(META, 'readwrite');
      tx.objectStore(META).put(meta);
      await done(tx);
    } catch (e) {
      if (isQuota(e)) throw new StorageFullError('Browser storage is full.');
      throw e;
    }
  }

  async remove(id: string): Promise<void> {
    const db = await this.db;
    const tx = db.transaction([META, FILES], 'readwrite');
    tx.objectStore(META).delete(id);
    tx.objectStore(FILES).delete(id);
    await done(tx);
  }
}

export async function storageEstimate(): Promise<{ usage: number; quota: number } | null> {
  try {
    const e = await navigator.storage?.estimate?.();
    if (e && e.quota) return { usage: e.usage ?? 0, quota: e.quota };
  } catch {
    /* ignore */
  }
  return null;
}

export async function requestPersistence(): Promise<boolean> {
  try {
    if (navigator.storage?.persisted && (await navigator.storage.persisted())) return true;
    return (await navigator.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

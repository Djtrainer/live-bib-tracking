// The operator login, as the browser sees it.
//
// The credentials are checked by the API (src/api_backend/admin_auth.py);
// nothing in this bundle knows them. A successful login sets an HttpOnly
// session cookie, which the browser attaches to same-origin requests by
// itself, so the calls below never touch a token. What they add is one
// rule: an operator action answered 401 means the session is gone, and the
// operator is sent back to the login with the reason.

export interface AdminSession {
  /** A login is configured on the server. False = the API runs open. */
  configured: boolean;
  /** This browser may use Live Management (always true when not configured). */
  authenticated: boolean;
  /** The server could not be asked; do not bounce the operator on a network blip. */
  unknown?: boolean;
}

export async function getAdminSession(): Promise<AdminSession> {
  try {
    const res = await fetch('/api/admin/session', { credentials: 'same-origin' });
    if (res.status === 404) return { configured: false, authenticated: true };   // an older API without the login
    if (!res.ok) return { configured: true, authenticated: false, unknown: true };
    const body = await res.json();
    return { configured: !!body.configured, authenticated: !!body.authenticated };
  } catch {
    return { configured: true, authenticated: false, unknown: true };
  }
}

export async function adminLogin(
  username: string,
  password: string,
): Promise<{ ok: boolean; configured: boolean; message?: string }> {
  try {
    const res = await fetch('/api/admin/login', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (res.status === 401) return { ok: false, configured: true, message: 'Invalid username or password. Please try again.' };
    if (!res.ok) return { ok: false, configured: true, message: `The server answered ${res.status}. Is the results API running?` };
    const body = await res.json();
    return { ok: true, configured: body.configured !== false };
  } catch {
    return { ok: false, configured: true, message: 'Could not reach the results API.' };
  }
}

export async function adminLogout(): Promise<void> {
  try {
    await fetch('/api/admin/logout', { method: 'POST', credentials: 'same-origin' });
  } catch {
    /* the cookie expires on its own; signing out must not fail on a dead API */
  }
}

/** fetch() for operator actions: a 401 sends the operator back to the login. */
export async function adminFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const res = await fetch(input, { credentials: 'same-origin', ...init });
  if (res.status === 401) {
    window.location.assign('/admin?expired=1');
    throw new Error('Admin login required');
  }
  return res;
}

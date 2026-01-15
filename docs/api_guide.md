# Example API Guide

This is a tiny, fake API reference used to test a documentation RAG pipeline.

## Authentication

To authenticate, you **log in** with an API key.

### Login

Send a POST request to `/api/v1/login` with your API key.

```bash
curl -X POST https://api.example.com/api/v1/login \
  -H 'Authorization: Bearer <API_KEY>'
```

If the key is valid, the API returns a session token.

### Reset API Key

To reset an API key, send a POST request to `/api/v1/reset`.

```bash
curl -X POST https://api.example.com/api/v1/reset \
  -H 'Authorization: Bearer <API_KEY>'
```

## Errors

### ERR-AD-99

`ERR-AD-99` means the API key is expired.

**Fix:** reset the key using `/api/v1/reset`, then log in again.

### ERR-NONSYNC-109

`ERR-NONSYNC-109` means the account is out of sync.

**Fix:** wait 60 seconds and retry. If it persists, contact support with the request id.

## Security Best Practices

- API keys expire after **90 days**.
- Store keys in a secret manager.
- Do not hardcode keys in source code.

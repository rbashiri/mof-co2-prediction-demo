# Team + Contact section integration (React / TypeScript)

1. Copy `TeamContact.tsx` and `TeamContact.css` into `frontend/src/`.
2. Copy `csu-chico-campus.png` into `frontend/public/` (not `src`).
3. At the top of `frontend/src/App.tsx`, add:
   `import TeamContact from './TeamContact'`
4. Near the end of the App JSX, immediately before the closing `</div>` for `<div className="container">`, insert:
   `<TeamContact />`
   Put it **outside** the `{result && (...)}` conditional so it is visible before prediction.
5. Confirm the email in `TeamContact.tsx` is correct before publishing. Add teammates' real LinkedIn/GitHub URLs only after confirming them.
6. Run `npm run build` from `frontend/` and check desktop/mobile layouts.

The form uses `mailto:` and **does not send email automatically**. Visitors must send the composed email in their own mail client. For direct web form submission, connect an email backend/service later.

The university image is the user-provided photo; the code does not use generated or stock people portraits. The component is standalone and does not change prediction or Gemini code.

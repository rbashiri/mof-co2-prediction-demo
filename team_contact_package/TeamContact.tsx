import { useState, type FormEvent } from 'react'
import './TeamContact.css'

type Member = { initials: string; name: string; role: string; expertise: string; organization: string; secondary?: string; location: string; github?: string; linkedin?: string }

const members: Member[] = [
  { initials: 'RB', name: 'Robabeh Bashiri, PhD', role: 'Project Lead', expertise: 'Materials Scientist · Data Scientist', organization: 'California State University, Chico', secondary: 'Department of Chemistry and Biochemistry', location: 'Chico, California, USA', github: 'https://github.com/rbashiri' },
  { initials: 'RK', name: 'Ripunjay Kumar', role: 'Team Member', expertise: 'Data Scientist · Machine Learning', organization: 'Syngenta Group', secondary: 'Indian Institute of Technology Madras', location: 'Pune, India' },
  { initials: 'UF', name: 'Umer Farooq', role: 'Team Member', expertise: 'Materials Research · Energy Storage', organization: 'Bahauddin Zakariya University', secondary: 'Undergraduate Student · Volunteer Researcher', location: 'Multan, Pakistan' },
]

const contactEmail = 'robabeh.bashiri@gmail.com' // Verify before publishing

export default function TeamContact() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const subject = encodeURIComponent(`MOF CO₂ research inquiry from ${name}`)
    const body = encodeURIComponent(`Name: ${name}\nEmail: ${email}\n\n${message}`)
    // Opens the visitor's mail application; does not transmit or store the message itself.
    window.location.href = `mailto:${contactEmail}?subject=${subject}&body=${body}`
  }

  return (
    <section className="team-contact" aria-label="Research team and contact information">
      <div className="tc-heading">
        <span className="tc-eyebrow">OUR COLLABORATORS</span>
        <h2>Meet Our Team</h2>
        <p>Bringing together materials science, machine learning, and scientific interpretation.</p>
      </div>
      <div className="tc-members">
        {members.map((person) => (
          <article className="tc-member" key={person.initials}>
            <div className="tc-avatar" aria-hidden="true">{person.initials}</div>
            <span className="tc-role">{person.role}</span>
            <h3>{person.name}</h3>
            <p className="tc-expertise">{person.expertise}</p>
            <div className="tc-divider" />
            <p className="tc-org">{person.organization}</p>
            {person.secondary && <p className="tc-secondary">{person.secondary}</p>}
            <p className="tc-location">⌖ {person.location}</p>
            {person.github && <a href={person.github} target="_blank" rel="noopener noreferrer">GitHub ↗</a>}
            {person.linkedin && <a href={person.linkedin} target="_blank" rel="noopener noreferrer">LinkedIn ↗</a>}
          </article>
        ))}
      </div>

      <section className="tc-contact" aria-labelledby="tc-contact-title">
        <div className="tc-campus">
          <img src="/csu-chico-campus.png" alt="California State University, Chico campus" />
          <div className="tc-campus-info">
            <span className="tc-eyebrow">PROJECT LEAD'S AFFILIATION</span>
            <h3>California State University, Chico</h3>
            <p>Department of Chemistry and Biochemistry</p>
            <p>Chico, California, USA</p>
            <p className="tc-campus-note">Academic affiliation of the project lead; not a claim of institutional sponsorship.</p>
          </div>
        </div>
        <div className="tc-form-panel">
          <span className="tc-eyebrow">LET'S COLLABORATE</span>
          <h2 id="tc-contact-title">Contact Us</h2>
          <p>Have a research question, collaboration idea, or feedback about the MOF CO₂ Predictor? We'd love to hear from you.</p>
          <form onSubmit={handleSubmit}>
            <label htmlFor="tc-name">Your name</label>
            <input id="tc-name" type="text" value={name} onChange={e => setName(e.target.value)} required maxLength={100} />
            <label htmlFor="tc-email">Your email</label>
            <input id="tc-email" type="email" value={email} onChange={e => setEmail(e.target.value)} required maxLength={200} />
            <label htmlFor="tc-message">Your message</label>
            <textarea id="tc-message" rows={4} value={message} onChange={e => setMessage(e.target.value)} required maxLength={3000} />
            <button type="submit">Compose Email ↗</button>
            <p className="tc-hint">Opens your email app; you must press Send there. No message is submitted by this website.</p>
          </form>
          <div className="tc-links">
            <a href={`mailto:${contactEmail}`}>✉ Email</a>
            <a href="https://github.com/rbashiri" target="_blank" rel="noopener noreferrer">⌘ GitHub</a>
          </div>
        </div>
      </section>
    </section>
  )
}

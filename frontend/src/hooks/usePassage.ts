import { useState } from "react"
import { apiPassage } from "../api/router"

interface UsePassageData {
    status: string | undefined
    error: string | undefined
    getStudyNotes : (reference: string) => Promise<string>
    passageText: string
    icebreaker: string
    context: string[]
    questions: string[]
    lifeApplication: string[]
}

export function usePassage(): UsePassageData {

    const [passageText, setPassageText] = useState<string>("")
    const [icebreaker, setIcebreaker] = useState<string>("")
    const [context, setContext] = useState<string>("")
    const [questions, setQuestions] = useState<string>("")
    const [lifeApplication, setLifeApplication] = useState<string>("")
    const [status, setStatus] = useState<string>()
    const [error, setError] = useState<string>()

    async function getStudyNotes(reference: string): Promise<string> {
        setStatus("loading")
        setError(undefined)
        return apiPassage(reference).then((response) => {
            const sections = response.split("###").map(section => section.trim())
            setPassageText(sections[0] || "")
            setIcebreaker(sections[1] || "")
            setContext(sections[2] || "")
            setQuestions(sections[3] || "")
            setLifeApplication(sections[4] || "")
            setStatus("success")
            return response
        }).catch((err) => {
            setError(err.message || "An error occurred")
            setStatus("error")
            return Promise.reject(err)
        })
    }

    return {
        status,
        error,
        getStudyNotes,
        passageText,
        icebreaker,
        context,
        questions,
        lifeApplication
    }
} 
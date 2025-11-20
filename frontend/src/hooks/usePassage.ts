import { useState } from "react"
import { apiPassage, type PassageResponse } from "../api/router"

interface UsePassageData {
    status: string | undefined
    error: string | undefined
    getStudyNotes : (reference: string) => Promise<PassageResponse>
    passageText: string
    icebreaker: string
    context: string[]
    questions: string[]
    lifeApplication: string[]
}

export function usePassage(): UsePassageData {

    const [passageText, setPassageText] = useState<string>("")
    const [icebreaker, setIcebreaker] = useState<string>("")
    const [context, setContext] = useState<string[]>([])
    const [questions, setQuestions] = useState<string[]>([])
    const [lifeApplication, setLifeApplication] = useState<string[]>([])
    const [status, setStatus] = useState<string>()
    const [error, setError] = useState<string>()

    async function getStudyNotes(reference: string): Promise<PassageResponse> {
        setStatus("loading")
        setError(undefined)
        return apiPassage(reference).then((response) => {
            console.log("API Response:", response)
            const studyNotes = response.study_notes
            const sections = studyNotes.split("###").map(section => section.trim())
            const rawContext = sections[2] ? sections[2].split("\n").map(line => line.trim()).filter(line => line.length > 0) : []
            const rawQuestions = sections[3] ? sections[3].split("\n").map(line => line.trim()).filter(line => line.length > 0) : []
            const rawLifeApplication = sections[4] ? sections[4].split("\n").map(line => line.trim()).filter(line => line.length > 0) : []
            setPassageText(sections[0] || "")
            setIcebreaker(sections[1] || "")
            setContext(rawContext || [])
            setQuestions(rawQuestions || [])
            setLifeApplication(rawLifeApplication|| [])
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
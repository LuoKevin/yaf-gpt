import api from './client.ts'


export type PassageModel = {
 reference: string
}

export function apiPassage(reference: string): Promise<void> {
    return api.get("/passage", { params : {reference: reference} })
}